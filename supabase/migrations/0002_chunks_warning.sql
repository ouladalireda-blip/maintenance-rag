-- =====================================================================
-- 0002 — Passages contenant un « Attention ! » (has_warning)
-- 0001_init.sql est figé : ce fichier n'ajoute qu'une colonne et redéfinit
-- hybrid_search pour la renvoyer. Tout ou rien : une seule transaction.
-- =====================================================================

begin;

-- ---------------------------------------------------------------------
-- Colonne has_warning : vrai si le passage contient un avertissement
-- « Attention ! » (Chunk.has_warning, calculé au découpage). Les lignes
-- existantes prennent false ; la prochaine ingestion la recalcule.
-- ---------------------------------------------------------------------
alter table public.chunks
    add column has_warning boolean not null default false;

comment on column public.chunks.has_warning is
    'Vrai si le passage contient un avertissement « Attention ! » (calculé au découpage).';

-- ---------------------------------------------------------------------
-- hybrid_search : on ne peut pas changer le type de retour d'une fonction
-- avec « create or replace » ; on la supprime puis on la recrée.
-- Paramètres, valeurs par défaut et logique RRF IDENTIQUES à 0001_init.sql ;
-- seule nouveauté : la colonne has_warning, ajoutée en fin de résultat.
-- ---------------------------------------------------------------------
drop function public.hybrid_search(
    text, extensions.vector, text[], int, int, float, float, float, text
);

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
    has_warning         boolean  -- nouveau (0002) : le passage contient un « Attention ! »
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
            select c.id, ts_rank_cd(c.fts, websearch_to_tsquery('simple', query_text)) as score
            from chunks c
            where c.document_id in (select a.id from allowed_documents a)
              and c.fts @@ websearch_to_tsquery('simple', query_text)
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
-- Droits : une fonction recréée reçoit les droits par défaut (exécution
-- pour public) ; on les retire de nouveau. Seul le backend (service_role,
-- clé jamais exposée au navigateur) peut l'appeler via l'API (RPC).
-- ---------------------------------------------------------------------
revoke execute on function public.hybrid_search(
    text, extensions.vector, text[], int, int, float, float, float, text
) from public, anon, authenticated;

grant execute on function public.hybrid_search(
    text, extensions.vector, text[], int, int, float, float, float, text
) to service_role;

-- ---------------------------------------------------------------------
-- Contrôle : la RLS doit rester active sur les deux tables (activée en
-- 0001, sans politique = aucun accès avec la clé publique). Sinon, la
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
