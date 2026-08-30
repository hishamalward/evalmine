# @evalmine/harness-kit

Producer-side helpers for completed
[EvalMine external-artifact bundles](../../docs/spec.md#13g-completed-external-artifact-import).
The package has no runtime dependencies and no model, judge, ledger, or report access.

```ts
import {
  HarnessBuilder,
  correlationId,
  sharedPrompt,
  writeBundle,
} from "@evalmine/harness-kit";

const builder = new HarnessBuilder({
  external_artifacts: "prompt-bakeoff",
  question: "Which completed condition is strongest?",
  evaluation: {
    objectives: ["Correctness", "Usefulness"],
    blind: "condition",
    ranking_style: "n-way",
    fields: ["summary"],
    human: { required: true, labels_per_pair: 1, coverage: "calibration-subset" },
    judge: {
      enabled: true,
      pairwise: false,
      position_swap: false,
      calibrate: true,
      runner: "api-prompt",
      model: "provider/judge-model",
    },
  },
});

const prompt = sharedPrompt({
  input: { document: "synthetic input" },
  whatVaries: "Only the declared condition changes.",
});

builder.addLayer(completedRecordsFromRunOne);
builder.addLayer(completedRecordsFromRunTwo); // same block + condition replaces run one

const bundle = builder.build();
await writeBundle(bundle, "/tmp/prompt-bakeoff");
```

`item_id` must describe the item being compared, independent of the condition. The
comparison block is `(lane, item_id, account_id)`. Every record in that block must have
the exact same shared prompt, and the prompt should contain the real common input but no
rendered arm-specific instructions. These meanings cannot be inferred from opaque strings;
the kit enforces their structural consequences and leaves their semantic truth to the
producer.
