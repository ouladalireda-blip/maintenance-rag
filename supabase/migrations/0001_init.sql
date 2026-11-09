-- =====================================================================
-- Assistant documentaire maintenance — schéma initial (v2)
-- Documents → passages indexés trois fois : vecteur, plein texte, références exactes.
-- =====================================================================

create extension if not exists vector with schema extensions;

-- ---------------------------------------------------------------------
-- Documents sources : traçabilité et licence de chaque fichier du corpus.
-- ---------------------------------------------------------------------
create table public.documents (
    id           uuid primary key default gen_random_uuid(),
    slug         text not null unique,           -- = id du manifeste (corpus/manifest.yaml), ex. "inrs-ed6109"
    title        text not null,
    source_url   text,
    license      text not null,                  -- ex. "Domaine public (OSHA)", "Fictif (projet)"
    language     text not null check (language in ('fr', 'en')),
    version      text,
    machine      text,                           -- ex. "TX-200" ; null = document général (sécurité)
    page_offset  int  not null default 0 check (page_offset >= 0),  -- page imprimée = page PDF − page_offset
    sha256       text not null unique,           -- ingestion idempotente : un fichier n'est indexé qu'une fois
    ingested_at  timestamptz not null default now()
);

comment on column public.documents.slug is
    'Identifiant stable du document, égal au champ id de corpus/manifest.yaml.';
comment on column public.documents.page_offset is
    'Page imprimée = page PDF - page_offset ; une page dont le résultat est <= 0 (couverture) n''a pas de page imprimée.';

-- ---------------------------------------------------------------------
-- Passages : l'unité récupérée, re-classée et citée.
-- Les métadonnées de citation sont obligatoires dès l'indexation.
-- page_start / page_end sont des numéros de page PDF (1 = première page du fichier),
-- PAS des pages imprimées : celles-ci sont calculées avec documents.page_offset.
-- ---------------------------------------------------------------------
create table public.chunks (
    id             bigint generated always as identity primary key,
    document_id    uuid not null references public.documents (id) on delete cascade,
    chunk_index    int  not null check (chunk_index >= 0),
    section_title  text not null,
    page_start     int  not null check (page_start >= 1),          -- page PDF
    page_end       int  not null check (page_end >= page_start),   -- page PDF
    content        text not null check (length(content) > 0),
    token_count    int  not null check (token_count > 0),
    identifiers    text[] not null default '{}', -- références exactes extraites (ERR-402, FILTRE-XJ402)
    embedding      extensions.vector(1536) not null,     -- 1024 si bge-m3
    -- Plein texte calculé par la base, configuration 'simple' : ni racinisation
    -- ni mots vides, pour un comportement identique en français et en anglais.
    fts            tsvector generated always as (
                       to_tsvector('simple', section_title || ' ' || content)
                   ) stored,
    unique (document_id, chunk_index)
);

comment on column public.chunks.page_start is
    'Numéro de page PDF (>= 1), pas la page imprimée : voir documents.page_offset.';
comment on column public.chunks.page_end is
    'Numéro de page PDF (>= page_start), pas la page imprimée : voir documents.page_offset.';

create index chunks_embedding_hnsw on public.chunks
    using hnsw (embedding extensions.vector_cosine_ops);     -- recherche vectorielle approximative
create index chunks_fts_gin         on public.chunks using gin (fts);          -- plein texte
create index chunks_identifiers_gin on public.chunks using gin (identifiers);   -- références exactes
create index chunks_document_id     on public.chunks (document_id);

-- ---------------------------------------------------------------------
-- Sécurité : RLS activée sans politique = aucun accès avec la clé publique.
-- Seul le backend (clé de service, jamais exposée au navigateur) lit et écrit.
-- ---------------------------------------------------------------------
alter table public.documents enable row level security;
alter table public.chunks    enable row level security;

-- ---------------------------------------------------------------------
-- Recherche hybride à trois classements, fusionnés par RRF.
-- Les rangs de chaque méthode sont renvoyés : ils servent à l'évaluation
-- (quelle méthode a trouvé le bon passage ?) et au débogage.
-- Chaque classement sélectionne d'abord ses candidats par une sous-requête
-- « order by … limit » (compatible avec l'index HNSW pour le vecteur), puis
-- numérote le résultat. L'id départage les ex-aequo : résultats déterministes.
-- ---------------------------------------------------------------------
create or replace function public.hybrid_search(
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
    rrf_score           float
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
           f.semantic_rank, f.full_text_rank, f.identifier_rank, f.rrf_score
    from fused f
    join chunks c    on c.id = f.id
    join documents d on d.id = c.document_id
    order by f.rrf_score desc, c.id
    limit match_count;
$$;

-- Fonction réservée au backend : ni la clé publique (anon) ni les utilisateurs
-- authentifiés ne peuvent l'appeler via l'API (RPC).
revoke execute on function public.hybrid_search(
    text, extensions.vector, text[], int, int, float, float, float, text
) from public, anon, authenticated;
