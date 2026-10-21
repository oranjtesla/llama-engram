# JEV classifier in llama-server

`--jev-dir` enables JEV's trained decision head on the existing `/v1/systemone`
endpoint. The backing Gemma 4 StyleTune model remains shared with roleplay and
maintenance requests. No other model process is required.

Example server options (keep the existing inference/thread/KV settings):

```sh
llama-server \
  -m /models/styletune.gguf \
  --lora-scaled /models/jev-lora.gguf:0 \
  --lora-init-without-apply \
  --jev-dir /models/jev \
  --slot-save-path /srv/slots/ \
  --kv-unified -np 5
```

`/models/jev` must contain the materialized `head.safetensors`,
`judge_config.json`, and `calibration_gold.json` files. `head.safetensors` is
tracked by Git LFS and must not be left as a pointer file. The adapter is a
separate GGUF loaded through the ordinary `--lora-scaled` option.

For a classification request:

```json
{
  "id_slot": 3,
  "lora": [{"id": 0, "scale": 1.0}],
  "state": "<narrative transcript>",
  "questions": {
    "school_morning": {
      "type": "noul",
      "instructions": "Is the current scene taking place at school during morning hours, before the end of homeroom?"
    }
  }
}
```

Send this object to `POST /v1/systemone`. The response uses the existing
`answers.<question_id>.noul` probability-of-true field. The endpoint also
supports `choice` (2–255 named criteria) and `score` (2–10 ordered levels).
These can be mixed with Noul questions and all accept structured JSON in
`state`, `instructions`, and criteria descriptions. The reference classifier
uses LoRA scale 1.0; changing it may invalidate the trained calibration.

### Decision rendering

The default `tools/server/jev-bare-v1.jinja` is embedded in the server at
build time. It renders plain-string Noul requests byte-identically to JEV's
reference `bare-v1` prompt, including the `false`/`true` canonical options.
Structured state, instructions, and criteria descriptions are passed as JSON
values to Jinja: **the template is the sole owner of their prompt formatting.**
The default template includes descriptive criteria without replacing canonical
Noul/Score output labels. This is a local compatibility convention, not a
reproduction of TypeSafe's private hosted prompt format.

Override it without recompiling via `--decision-template-file /path/to/file.jinja`
(or `LLAMA_ARG_DECISION_TEMPLATE_FILE`). This option is independent of
`--chat-template-file`. The template context contains `id`, `type`, `state`,
`instructions`, `options` (`key` and structured `description`), and `images`.
Custom templates must retain a final readout position compatible with the
model's trained last-token head.

### Output capacity

The trained head has two Noul outputs, six Score outputs (0–5), and sixteen
Choice outputs. A six-level Score uses its native head. Other Score lengths
(2–5, 7–10) are evaluated using the Choice head and presented in the standard
Score response shape; this is a *compatibility approximation*, not the model's
native Score calibration. Choices with more than sixteen options run in
balanced groups, followed by a comparison of group winners. Relative group
odds are combined with the final comparison to retain every option. Wide
Choice results require further full-model evaluation before they should be
used as calibrated confidence gates.


`id_slot` is optional. When supplied, multiple questions in the same request
are queued for that slot. Engram can distribute questions concurrently by
sending separate requests with different `id_slot` values. Unpinned decision
requests use the normal server task scheduler and shared-prefix grouping.

Ordinary RP requests omit `lora` and use the scale-zero server default.
A slot's cached KV is invalidated when its adapter configuration changes. RAM
prompt-cache entries are also matched by adapter configuration.

When `--jev-dir` is enabled, `POST /slots/<id>?action=save` saves an adjacent
`<filename>.lora.json` compatibility record for each slot snapshot. Restore
checks this record against the snapshot and the loaded adapters. Keep the
sidecar and slot snapshot together. A snapshot without compatible provenance
is rejected to avoid using KV generated under the wrong LoRA. Slot selection,
slot snapshots, and coordination between restore/inference/save remain owned
by Engram.

## Validation

`tests/test-jev-head.cpp` evaluates `dev/jev/reference/e1.json` against the
actual JEV head for all three classifier ranges. Noul expects probability of
true approximately 0.743557. `tests/test-jev-template.cpp` checks plain Noul
prompt parity and preservation of structured data in the default Jinja template.
`tests/test-slot-lora.cpp` tests snapshot adapter provenance, including
mismatching adapter paths and modified snapshot files.

Full-model hidden-state equivalence and the eight narrative predicate
questions require the StyleTune GGUF and JEV LoRA loaded on the inference host.
