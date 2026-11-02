# American English IPA for ASR

[ipa-dict](https://github.com/open-dict-data/ipa-dict) is a set of monolingual word lists with IPA pronunciations. The US English list is General American, and each row is already `word<TAB>IPA`.

This checkout is the `master` tree at `972cd67` (22 October 2026):

```text
ipa-dict/data/en_US.txt
```

`en_US.txt` is UTF-8, 3,180,267 bytes, and 125,927 entries. Headwords are lowercase. Inflected forms are separate entries (`run`, `ran`, `runs`, `running`).

The repository is MIT-licensed. Third-party lists keep their own licenses; the upstream [Credits](https://github.com/open-dict-data/ipa-dict#credits) section records those. US English is based on [cmudict-ipa](https://github.com/lingz/cmudict-ipa) (MIT).

## Download

Full repository, if GitHub CLI is installed:

```bash
gh repo clone open-dict-data/ipa-dict
cd ipa-dict
```

American English is then at `data/en_US.txt`.

Only that file:

```bash
gh api \
  repos/open-dict-data/ipa-dict/contents/data/en_US.txt \
  --jq '.content' \
  | base64 --decode > en_US.txt
```

Preview and search:

```bash
head en_US.txt
grep -i '^canadian	' en_US.txt
grep -i '^canada	' en_US.txt
```

On this machine the same lookups are:

```powershell
Get-Content "ipa-dict\data\en_US.txt" -TotalCount 15
Select-String -Path "ipa-dict\data\en_US.txt" -Pattern "^canadian`t"
Select-String -Path "ipa-dict\data\en_US.txt" -Pattern "^canada`t"
```

Release [1.0](https://github.com/open-dict-data/ipa-dict/releases/tag/1.0) (20 September 2016) also publishes JSON, CSV, and XML. The tab-delimited files in `ipa-dict/data/` are the current lists. Prefer those when a release archive and this checkout disagree.

## File format

Each line is:

```text
WORD<TAB>IPA
```

Every pronunciation is wrapped in `/ /`. When a spelling has more than one pronunciation, the alternatives are separated by a comma and a space.

Checked rows from `ipa-dict/data/en_US.txt`:

```text
canada      /ˈkænədə/
canadian    /kəˈneɪdiən/
computer    /kəmˈpjutɝ/
read        /ˈɹɛd/, /ˈɹid/
the         /ˈðə/, /ðə/, /ði/
```

This transcription uses `ɡ` (U+0261), `ɹ`, `ɫ`, and `ɝ`. It does not use a length mark (`ː`) or `ɚ`.

## Load it in Python

```python
from pathlib import Path

ipa_dict = {}
path = Path("ipa-dict/data/en_US.txt")

with path.open(encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        word, ipa = line.split("\t", 1)
        ipa_dict[word.lower()] = ipa

print(ipa_dict.get("canadian"))  # /kəˈneɪdiən/
print(ipa_dict.get("canada"))    # /ˈkænədə/
```

ASR preprocessing:

```python
word = "canadian"
ipa = ipa_dict.get(word.lower())
if ipa:
    print("IPA:", ipa)
else:
    print("Word not found")
```

Keys in the file are already lowercase, so lowercase the incoming word before lookup.

## Convert it to JSON

A flat object is the convenient form for lookup. Upstream release JSON is wrapped (`{"en_US": [ { "word": "ipa", ... } ]}`). This script writes the flat object instead.

```python
import json
from pathlib import Path

ipa_dict = {}
with Path("ipa-dict/data/en_US.txt").open(encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if "\t" not in line:
            continue
        word, ipa = line.split("\t", 1)
        ipa_dict[word] = ipa

with Path("en_US.json").open("w", encoding="utf-8") as f:
    json.dump(ipa_dict, f, ensure_ascii=False, indent=2)
```

```json
{
  "canada": "/ˈkænədə/",
  "canadian": "/kəˈneɪdiən/"
}
```

## Other languages

The same `word<TAB>IPA` layout is used for every file in `ipa-dict/data/`.

| File | Language |
| --- | --- |
| `en_US.txt` | English, General American |
| `en_UK.txt` | English, Received Pronunciation |
| `de.txt` | German |
| `fr_FR.txt` | French (France) |
| `fr_QC.txt` | French (Québec) |
| `es_ES.txt` | Spanish (Spain) |
| `es_MX.txt` | Spanish (Mexico) |
| `pt_BR.txt` | Portuguese (Brazil) |
| `ja.txt` | Japanese |
| `ko.txt` | Korean |
| `zh_hans.txt` | Mandarin, simplified characters |
| `zh_hant.txt` | Mandarin, traditional characters |
| `yue.txt` | Cantonese |
| `vi_N.txt` | Vietnamese (Northern) |
| `vi_C.txt` | Vietnamese (Central) |
| `vi_S.txt` | Vietnamese (Southern) |
| `ar.txt` | Arabic (Modern Standard) |
| `fa.txt` | Persian |
| `nl.txt` | Dutch |
| `sv.txt` | Swedish |
| `sw.txt` | Swahili |
| `eo.txt` | Esperanto |
| `fi.txt` | Finnish |
| `is.txt` | Icelandic |
| `jam.txt` | Jamaican Creole |
| `km.txt` | Khmer |
| `ma.txt` | Malay (Malaysian and Indonesian) |
| `nb.txt` | Norwegian Bokmål |
| `or.txt` | Odia |
| `ro.txt` | Romanian |
| `tts.txt` | Isan |

Mandarin is `zh_hans.txt` and `zh_hant.txt`. Both files share pronunciations; the suffix is the orthography.

## CTC targets

How those tokens are trained and decoded is written up in [docs/asr.md](docs/asr.md), with the CTC, lexicon-search, and IPA phone-recognition papers.

Keep `en_US.txt` as the pronunciation lexicon. For an encoder + CTC model, the label sequence is the individual IPA phonemes. The whole pronunciation string is one label per word, so the vocabulary cannot share phones across words and every unseen spelling is a new class.

Split each field into alternatives, strip the slashes, then longest-match the phones below. In this file, `a`, `e`, and `o` occur only inside diphthongs. `i` and `u` also occur alone (`/ˈdi/`, `/ˈkjuz/`). Affricates and diphthongs are two code points and should stay one token: `tʃ`, `dʒ`, `aɪ`, `aʊ`, `eɪ`, `oʊ`, `ɔɪ`.

| Group | Tokens |
| --- | --- |
| Stress | `ˈ` `ˌ` |
| Affricates | `tʃ` `dʒ` |
| Diphthongs | `aɪ` `aʊ` `eɪ` `oʊ` `ɔɪ` |
| Vowels | `i` `u` `ɪ` `ɛ` `æ` `ɑ` `ɔ` `ʊ` `ə` `ɝ` |
| Consonants | `b` `d` `f` `h` `j` `k` `m` `n` `p` `s` `t` `v` `w` `z` `ð` `θ` `ʃ` `ʒ` `ŋ` `ɡ` `ɫ` `ɹ` |

`ˈ` and `ˌ` are their own tokens, so stress stays in the target. Drop them only if the model should predict segments without stress.

A CTC target is one token sequence. Entries such as `read` and `the` list several pronunciations; take the one that matches the utterance. The first listed form is a reasonable default when the reading is unknown.

```python
PHONEMES = [
    "tʃ", "dʒ",
    "aɪ", "aʊ", "eɪ", "oʊ", "ɔɪ",
    "ɑ", "æ", "ɛ", "ɪ", "ɔ", "ʊ", "ə", "ɝ", "i", "u",
    "b", "d", "f", "h", "j", "k", "m", "n", "p", "s", "t", "v", "w", "z",
    "ð", "ŋ", "ɡ", "ɫ", "ɹ", "ʃ", "θ", "ʒ",
    "ˈ", "ˌ",
]


def tokenize_ipa(ipa: str) -> list[str]:
    body = ipa.strip()
    if body.startswith("/") and body.endswith("/"):
        body = body[1:-1]
    tokens = []
    i = 0
    while i < len(body):
        match = next((phone for phone in PHONEMES if body.startswith(phone, i)), None)
        if match is None:
            raise ValueError(f"Unknown IPA symbol in {body[i:]!r}")
        tokens.append(match)
        i += len(match)
    return tokens


def pronunciations(field: str) -> list[list[str]]:
    return [tokenize_ipa(part) for part in field.split(",")]


# canadian -> ['k', 'ə', 'ˈ', 'n', 'eɪ', 'd', 'i', 'ə', 'n']
# canada   -> ['ˈ', 'k', 'æ', 'n', 'ə', 'd', 'ə']
# computer -> ['k', 'ə', 'm', 'ˈ', 'p', 'j', 'u', 't', 'ɝ']
```

That phone inventory covers every symbol in `en_US.txt`.
