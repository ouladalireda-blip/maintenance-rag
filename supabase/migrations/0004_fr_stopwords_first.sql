-- =====================================================================
-- 0004 — Mots vides retirés AVANT unaccent dans public.fr_unaccent
--
-- Cause mesurée (après 0003) : dans fr_unaccent, unaccent passait avant
-- french_stem. Un mot vide accentué perdait donc son accent avant d'être
-- comparé à la liste des mots vides français, qui ne le reconnaissait plus :
-- « à » → « a », « été » → « ete » restaient dans l'index et dans la
-- requête. Avec la requête en OU de hybrid_search, ces lexèmes parasites
-- faisaient matcher presque tout : 37 à 68 passages sur 69 par question, et
-- le MRR hybride baissait (0,759 → 0,657).
--
-- Correctif UNIQUEMENT de l'ordre des dictionnaires : un dictionnaire de mots
-- vides français passe en premier, puis unaccent, puis french_stem.
-- 0001 à 0003 sont figées. Tout ou rien : une seule transaction.
-- =====================================================================

begin;

-- ---------------------------------------------------------------------
-- Dictionnaire de mots vides : modèle simple (minuscules), liste
-- « french » de PostgreSQL, accept = false. Un mot vide est supprimé ;
-- tout autre mot n'est pas reconnu et passe au dictionnaire suivant
-- (unaccent puis french_stem). La liste contient les formes accentuées
-- (« à », « été », « où »…) : elle doit voir le mot AVANT unaccent.
-- ---------------------------------------------------------------------
create text search dictionary public.fr_stopwords (
    template  = pg_catalog.simple,
    stopwords = french,
    accept    = false
);

-- ---------------------------------------------------------------------
-- Nouvel ordre pour tous les types de mots, ASCII compris :
-- mots vides → unaccent → racinisation. Les mots ASCII passent aussi par
-- fr_stopwords pour qu'une même liste s'applique partout.
-- ---------------------------------------------------------------------
alter text search configuration public.fr_unaccent
    alter mapping for asciiword, asciihword, hword_asciipart, word, hword, hword_part
    with public.fr_stopwords, extensions.unaccent, french_stem;

-- ---------------------------------------------------------------------
-- fts reconstruite : modifier une configuration ne recalcule PAS une
-- colonne générée stockée ; les lexèmes déjà calculés (avec « a », « ete »)
-- resteraient. Index GIN et colonne supprimés, puis recréés avec
-- EXACTEMENT l'expression de 0003 : l'ajout de la colonne recalcule fts
-- pour toutes les lignes avec le nouvel ordre des dictionnaires.
--
-- hybrid_search n'est pas recréée : son corps est une chaîne ($$ … $$), sans
-- dépendance enregistrée envers chunks.fts, et elle ne désigne la
-- configuration que par son nom ('public.fr_unaccent'). Elle utilise donc
-- la nouvelle colonne et le nouvel ordre dès son prochain appel ; ses
-- droits (revoke / grant de 0003) sont inchangés.
-- ---------------------------------------------------------------------
drop index public.chunks_fts_gin;

alter table public.chunks drop column fts;

alter table public.chunks
    add column fts tsvector generated always as (
        to_tsvector('public.fr_unaccent', coalesce(section_title, '') || ' ' || coalesce(content, ''))
    ) stored;

create index chunks_fts_gin on public.chunks using gin (fts);

-- ---------------------------------------------------------------------
-- Contrôles : toute anomalie fait échouer la transaction, rien n'est
-- appliqué.
--   1. des mots vides accentués ne produisent plus aucun lexème ;
--   2. un mot avec ou sans accent donne le même lexème (unaccent actif) ;
--   3. la racinisation est toujours active (« consigner » → « consign ») ;
--   4. la RLS reste active sur documents et chunks.
-- ---------------------------------------------------------------------
do $$
begin
    if to_tsvector('public.fr_unaccent', 'à été était même') <> ''::tsvector then
        raise exception 'fr_unaccent garde des mots vides : migration annulée';
    end if;

    if to_tsvector('public.fr_unaccent', 'électrique')
       <> to_tsvector('public.fr_unaccent', 'electrique') then
        raise exception 'fr_unaccent ne retire plus les accents : migration annulée';
    end if;

    if not ('consign' = any (tsvector_to_array(to_tsvector('public.fr_unaccent', 'consigner')))) then
        raise exception 'fr_unaccent ne racinise plus : migration annulée';
    end if;

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
