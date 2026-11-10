# Corpus

Le manifeste de référence est [`corpus/manifest.yaml`](../corpus/manifest.yaml). Ce tableau le résume.

| Document | Source | Licence | Langue | Versionné |
|---|---|---|---|---|
| `inrs-ed6109` — INRS ED 6109, *Consignations et déconsignations* (oct. 2020) | INRS (inrs.fr) | © INRS 2020, reproduction interdite sans consentement : usage local uniquement | fr | **Non** (`corpus/inrs/` est dans `.gitignore`) |
| `tx200-*` — 4 procédures de la machine fictive TX-200 (à rédiger) | Projet | Fictif (projet) | fr | Oui (`corpus/tx200/`) |
| OSHA (facultatif, plus tard) | osha.gov | Domaine public (US) | en | À décider |

## Obtenir le PDF INRS ED 6109

1. Télécharger la brochure ED 6109 depuis le site officiel de l'INRS (rechercher « ED 6109 » sur inrs.fr), puis reporter l'URL exacte dans `source_url` du manifeste.
2. L'enregistrer sous `corpus/inrs/ed6109.pdf`.
3. Vérifier qu'il est bien ignoré par git :

   ```sh
   git check-ignore -v corpus/inrs/ed6109.pdf
   # attendu : .gitignore:<n>:corpus/inrs/	corpus/inrs/ed6109.pdf
   ```

## Vérifier le SHA-256

```sh
# Linux / macOS / Git Bash
sha256sum corpus/inrs/ed6109.pdf
```

```powershell
# PowerShell
(Get-FileHash corpus/inrs/ed6109.pdf -Algorithm SHA256).Hash.ToLower()
```

- Première fois : reporter la valeur dans `sha256` du manifeste.
- Ensuite : la valeur doit être identique à celle du manifeste. Si elle diffère, ce n'est pas la même édition : les numéros de page des questions `gold` ne sont plus fiables.

## Numérotation des pages

- **En base, les pages sont des pages PDF.** `chunks.page_start` et `chunks.page_end` valent 1 pour la première page du fichier, quelle que soit la numérotation imprimée. C'est la seule numérotation non ambiguë au moment de l'extraction.
- **Les citations utilisent la page imprimée**, calculée avec le `page_offset` du document (manifeste → `documents.page_offset`) : page imprimée = page PDF − `page_offset`. `hybrid_search` la renvoie dans `printed_page_start` et `printed_page_end`. Les questions `gold` des évaluations utilisent elles aussi la page imprimée (`printed_page`).
- **Les pages sans numéro imprimé** (couverture, pages liminaires) donnent un résultat ≤ 0 : la page imprimée vaut alors `NULL`, et une citation sur une telle page indique la page PDF.

Exemple pour l'ED 6109 (`page_offset: 2`) : les pages PDF 1 et 2 n'ont pas de page imprimée, et la page PDF 14 correspond à la page imprimée 12.

## Limites connues de l'extraction

Cas connus de `src/maintenance_rag/pdf_reader.py` sur l'ED 6109, documentés plutôt que corrigés. Chacun est suivi par un test.

| Cas | Où | Cause | Impact |
|---|---|---|---|
| Encadré rattaché au mauvais intertitre | p. 22 PDF (20 imprimée), chunk 44 | L'ordre de lecture lit la colonne gauche avant la droite. L'encadré en haut de la colonne droite passe donc après l'intertitre « Principes de séparation de circuits », placé plus bas dans la colonne gauche, et en hérite. | `section_title` trop précis (un niveau de trop). La recherche (vecteur, plein texte, identifiants `CLP`, `1272/2008`, `R4411-6`) et la page citée ne sont pas affectées. Test : `test_real_clp_box_section` (`xfail` strict). |
| Césures non recollées | « réa-lisent » (p. 10 PDF), « intempes-tivement » (p. 29 PDF) | `dehyphenate` ne recolle un mot coupé par un vrai tiret que si le mot entier apparaît ailleurs dans le document (vocabulaire construit sur le PDF). Ces deux mots n'y figurent qu'une fois, coupés. | Le texte affiché garde le tiret. Le garde-fou 2 n'est pas affecté (`normalize_for_match` supprime les tirets). La recherche plein texte peut manquer ces deux mots. |
