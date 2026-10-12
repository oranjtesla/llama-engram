# JEV narrative activation fixture

This fixture records the eight narrative predicate questions, their expected
answers, the prompt format used for JEV inference, and the known-good output for
question 1. The source narrative is `transcript.md`; the raw llama-server
`/embedding` result for question 1 is `e1.json` (one 2816-dimensional vector).

## Activation questions

| # | Activation question (`[question]`) | Expected | What it tests |
| --- | --- | --- | --- |
| 1 | Is the current scene taking place at school during morning hours, before the end of homeroom? | true | Scene continuity and time |
| 2 | Has Riku's scheduled meeting with Katsuragi-sensei in the staff room already taken place? | false | Future versus completed events |
| 3 | Does Asa currently feel responsible for Riku getting into trouble with their teacher? | true | Emotional inference and causality |
| 4 | Is Asa deliberately concealing her concern about Riku behind an outward appearance of indifference? | true | Internal state versus outward behavior |
| 5 | Does Asa know that Riku's mother is responsible for the cigarettes being in his bag? | false | Character-specific knowledge |
| 6 | Does Riku privately blame Asa for the incident despite recognizing that she isn't really at fault? | true | Contradictory feelings |
| 7 | Is Katsuragi-sensei still standing beside Riku's desk, directly confronting him about the cigarettes? | false | Superseded circumstances |
| 8 | Has Asa concluded that Riku is definitively a delinquent rather than possibly someone having a terrible first day? | false | Uncertainty versus established belief |

## Prompt format

The state is the ten-message narrative transcript in `transcript.md`. The
following is the provided example using question 5:

```text
[kind] noul
[state] <10-message transcript>
[question] Does Asa know that Riku's mother is responsible for the cigarettes being in his bag?
[options]
false
true
[decision]:
```

For each test case, substitute that case's activation question after
`[question]`. The source's literal field labels, option ordering and trailing
`[decision]:` punctuation are intentional.

## Reference result: question 1

`e1.json` contains the raw embedding vector returned for question 1. Applying
`jev_classify_embedding.py` with the reference classifier head and calibration
files yielded:

```json
{
  "false": 0.2564430236816406,
  "true": 0.7435570359230042,
  "decision": "true",
  "temperature": 1.2136210312191784,
  "hidden_state_l2_norm": 88.44002532958984
}
```

The vector is a reference fixture for classifier parity, not by itself proof
that another hidden-state extraction path produces the same representation.
