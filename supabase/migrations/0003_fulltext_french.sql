-- =====================================================================
-- 0003 — Plein texte en français : configuration fr_unaccent et requête en OU
--
-- Constat (evals/results/retrieval_dev_*.json) : la branche full-text de
-- hybrid_search ne trouvait aucun passage. Deux causes :
--   1. websearch_to_tsquery fait un ET de tous les mots de la question : une
--      question en langage naturel (« quelles », « dois », « faire »…) ne
--      peut jamais être contenue en entier dans un passage ;
--   2. la configuration 'simple' ne retire pas les mots vides et ne
--      racinise pas : « consigner » ne rencontre jamais « consignation ».
-- Mesure en base : Recall@10 full-text 8/8 et MRR 0,48 avec OU + 'french'
-- + titres (3/8 avec OU + 'simple').
--
-- 0001 et 0002 sont figées. Tout ou rien : une seule transaction.
-- =====================================================================

begin;

-- ---------------------------------------------------------------------
-- Accents : unaccent retire les diacritiques (« électrique » = « electrique »).
-- Une question tapée sans accents retrouve ainsi le texte accentué du PDF.
-- ---------------------------------------------------------------------
create extension if not exists unaccent with schema extensions;

-- ---------------------------------------------------------------------
-- Configuration fr_unaccent : copie de la configuration française de
-- PostgreSQL (mots vides français retirés, racinisation snowball
-- « french_stem »), avec unaccent appliqué AVANT la racinisation aux mots
-- (word), mots composés (hword) et parties de mots composés (hword_part).
-- Elle sert aux deux côtés, passages et questions : mêmes lexèmes.
-- ---------------------------------------------------------------------
create text search configuration public.fr_unaccent (copy = pg_catalog.french);

alter text search configuration public.fr_unaccent
    alter mapping for hword, hword_part, word with extensions.unaccent, french_stem;

-- ---------------------------------------------------------------------
-- hybrid_search lit chunks.fts : on supprime la fonction avant de toucher
-- à la colonne, puis on la recrée plus bas (son type de retour est celui
-- de 0002, mais on garde l'ordre « drop puis create » de 0002).
-- ---------------------------------------------------------------------
drop function public.hybrid_search(
    text, extensions.vector, text[], int, int, float, float, float, text
);

-- ---------------------------------------------------------------------
-- chunks.fts recalculée avec fr_unaccent. Une colonne générée ne peut pas
-- changer d'expression : index GIN et colonne supprimés, puis recréés.
-- L'ajout de la colonne recalcule fts pour toutes les lignes existantes ;
-- rien n'est calculé en Python (une seule source de vérité : cette
-- expression). Le titre de section est indexé avec le contenu : il porte
-- souvent les mots du technicien (« consignation électrique »,
-- « déconsignation »), absents du corps du passage.
-- ---------------------------------------------------------------------
drop index public.chunks_fts_gin;

alter table public.chunks drop column fts;

alter table public.chunks
    add column fts tsvector generated always as (
        to_tsvector('public.fr_unaccent', coalesce(section_title, '') || ' ' || coalesce(content, ''))
    ) stored;

create index chunks_fts_gin on public.chunks using gin (fts);

-- ---------------------------------------------------------------------
-- hybrid_search recréée : signature, valeurs par défaut, colonnes de
-- retour (has_warning compris) et fusion RRF IDENTIQUES à 0002. Seule la
-- branche full-text change :
--   - la question passe par to_tsvector('public.fr_unaccent', …) : mots
--     vides retirés, accents retirés, racinisation — exactement comme les
--     passages ;
--   - ses lexèmes sont reliés par « | » (OU) au lieu d'un ET de tous les
--     mots ;
--   - la chaîne est convertie par ::tsquery et NON par to_tsquery : les
--     lexèmes sont déjà racinisés, to_tsquery les raciniserait une seconde
--     fois (« consign » → racine d'une racine) et ne retrouverait plus ceux
--     de fts. quote_literal protège chaque lexème (apostrophe doublée) ;
--   - aucun lexème (question faite de mots vides) : la requête vaut NULL,
--     la branche ne renvoie rien et ne lève aucune erreur ;
--   - classement : ts_rank_cd desc, puis id pour départager.
-- ---------------------------------------------------------------------
create function public.hybrid_search(
    query_text         text,
    query_embedding    extensions.vector(1536),          -- 1024 si bge-m3
    query_identifiers  text[] default '{}',
    match_count        int    default 10,
    rrf_k              int    default 50,
    semantic_weight    float  default 1.0,
    full_text_weight   float  default 1.0,
    identifier_weight  float  default 2.0,               -- une référence exacte prime
    filter_machine     text   default null
)
returns table (
    chunk_id            bigint,
    document_slug       text,
    document_title      text,
    chunk_index         int,
    section_title       text,
    page_start          int,     -- page PDF
    page_end            int,     -- page PDF
    printed_page_start  int,     -- page imprimée ; null si la page n'en a pas (couverture)
    printed_page_end    int,     -- page imprimée ; null si la page n'en a pas (couverture)
    content             text,
    semantic_rank       bigint,
    full_text_rank      bigint,
    identifier_rank     bigint,
    rrf_score           float,
    has_warning         boolean  -- le passage contient un « Attention ! » (0002)
)
language sql stable
set search_path = public, extensions
as $$
    with allowed_documents as (
        -- Filtre machine : les documents de la machine demandée + les documents généraux.
        select d.id
        from documents d
        where filter_machine is null or d.machine = filter_machine or d.machine is null
    ),
    full_text_query as (
        -- Lexèmes de la question (fr_unaccent), reliés par OU ; NULL si aucun lexème.
        select nullif(
                   array_to_string(
                       array(
                           select quote_literal(t.lexeme)
                           from unnest(to_tsvector('public.fr_unaccent', query_text)) t
                       ),
                       ' | '
                   ),
                   ''
               )::tsquery as query
    ),
    semantic as (
        select s.id, row_number() over (order by s.distance, s.id) as rank_ix
        from (
            select c.id, c.embedding <=> query_embedding as distance
            from chunks c
            where c.document_id in (select a.id from allowed_documents a)
            order by c.embedding <=> query_embedding
            limit match_count * 3
        ) s
    ),
    full_text as (
        select s.id, row_number() over (order by s.score desc, s.id) as rank_ix
        from (
            select c.id, ts_rank_cd(c.fts, q.query) as score
            from chunks c
            cross join full_text_query q
            where q.query is not null
              and c.document_id in (select a.id from allowed_documents a)
              and c.fts @@ q.query
            order by score desc, c.id
            limit match_count * 3
        ) s
    ),
    identifier_match as (
        -- Classement par nombre de références de la question présentes dans le passage.
        select s.id, row_number() over (order by s.overlap desc, s.id) as rank_ix
        from (
            select c.id,
                   cardinality(array(
                       select unnest(c.identifiers) intersect select unnest(query_identifiers)
                   )) as overlap
            from chunks c
            where c.document_id in (select a.id from allowed_documents a)
              and cardinality(query_identifiers) > 0
              and c.identifiers && query_identifiers
            order by overlap desc, c.id
            limit match_count * 3
        ) s
    ),
    fused as (
        select coalesce(se.id, ft.id, im.id) as id,
               se.rank_ix as semantic_rank,
               ft.rank_ix as full_text_rank,
               im.rank_ix as identifier_rank,
               coalesce(semantic_weight   / (rrf_k + se.rank_ix), 0.0)
             + coalesce(full_text_weight  / (rrf_k + ft.rank_ix), 0.0)
             + coalesce(identifier_weight / (rrf_k + im.rank_ix), 0.0) as rrf_score
        from semantic se
        full outer join full_text ft        on ft.id = se.id
        full outer join identifier_match im on im.id = coalesce(se.id, ft.id)
    )
    select c.id, d.slug, d.title, c.chunk_index, c.section_title, c.page_start, c.page_end,
           case when c.page_start - d.page_offset > 0 then c.page_start - d.page_offset end,
           case when c.page_end   - d.page_offset > 0 then c.page_end   - d.page_offset end,
           c.content,
           f.semantic_rank, f.full_text_rank, f.identifier_rank, f.rrf_score,
           c.has_warning
    from fused f
    join chunks c    on c.id = f.id
    join documents d on d.id = c.document_id
    order by f.rrf_score desc, c.id
    limit match_count;
$$;

-- ---------------------------------------------------------------------
-- Droits (comme en 0002) : une fonction recréée reçoit les droits par
-- défaut (exécution pour public) ; on les retire. Seul le backend
-- (service_role, clé jamais exposée au navigateur) peut l'appeler (RPC).
-- ---------------------------------------------------------------------
revoke execute on function public.hybrid_search(
    text, extensions.vector, text[], int, int, float, float, float, text
) from public, anon, authenticated;

grant execute on function public.hybrid_search(
    text, extensions.vector, text[], int, int, float, float, float, text
) to service_role;

-- ---------------------------------------------------------------------
-- Contrôle (comme en 0002) : la RLS doit rester active sur les deux
-- tables (sans politique = aucun accès avec la clé publique). Sinon, la
-- transaction échoue et rien n'est appliqué.
-- ---------------------------------------------------------------------
do $$
begin
    if exists (
        select 1
        from pg_class c
        join pg_namespace n on n.oid = c.relnamespace
        where n.nspname = 'public'
          and c.relname in ('documents', 'chunks')
          and not c.relrowsecurity
    ) then
        raise exception 'RLS désactivée sur public.documents ou public.chunks : migration annulée';
    end if;
end
$$;

commit;
