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
