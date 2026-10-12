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
`answers.<question_id>.noul` probability-of-true field. JEV currently supports
only `noul` (false/true). The reference classifier uses LoRA scale 1.0;
changing it may invalidate the trained calibration.

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
actual JEV head. It expects probability of true approximately 0.743557.
`tests/test-slot-lora.cpp` tests snapshot adapter provenance, including
mismatching adapter paths and modified snapshot files.

Full-model hidden-state equivalence and the eight narrative predicate
questions require the StyleTune GGUF and JEV LoRA loaded on the inference host.
