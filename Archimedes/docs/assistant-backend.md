# Design assistant — proposed backend

The ASSISTANT tab is currently **interface only**. It renders the conversation,
suggested prompts, and the action-card mechanism; it holds no model. Send it
anything and it replies with the exact JSON payload it *would* dispatch, so the
contract is visible before any of it is built.

This is what I'd build behind it.

## Shape of the problem

This is not a chatbot bolted onto a CAD app. It is a **tool-using agent over a
solver**, and the interesting design questions are all about grounding:

1. It must never invent numbers. Every quantitative claim has to come from a
   solve that actually ran.
2. It must be able to *act* — change a parameter, re-solve, run a sweep — and
   those actions must be reviewable before they take effect.
3. The context is small and highly structured (a study spec plus scalar
   results), so this is not a RAG problem. Don't build a vector store.

## Architecture

Add an `/assistant` route to the existing bridge — same process, same
`ws://127.0.0.1:8791`, new op codes. It already owns the solver and the study
schema, which is exactly the context the model needs.

```
workbench UI  --ws-->  bridge  --https-->  Claude API
                          |
                          +--> archimedes.fem.solve()   (tools)
                          +--> archimedes.spec.Study    (schema)
```

```
{"op":"ask", "id":..., "message":"...", "context":{...}, "history":[...]}
{"op":"say", "id":..., "delta":"..."}                    streaming tokens
{"op":"propose", "id":..., "actions":[{...}]}            action cards
{"op":"said", "id":..., "usage":{...}}                   turn complete
```

Use the Claude API with tool use — `claude-sonnet-5` is the right default here
(fast enough for interactive turns, strong enough for the numeric reasoning);
reach for `claude-opus-5` on multi-step design studies. Stream the response so
tokens land in the panel as they arrive.

## Tools

The model gets the solver, not a description of the solver. All of these
already exist as Python functions:

| Tool | Maps to | Notes |
| --- | --- | --- |
| `get_study()` | `Study.to_dict()` | current parameters |
| `get_results()` | last `fem.solve()` output | peak stress, `|U|`, energy, reactions, SF |
| `set_parameter(key, value)` | `_apply_overrides` | dotted key, validated against the schema |
| `solve(mesher, modes)` | `fem.solve()` | the expensive one — needs confirmation |
| `sweep(param, values)` | `cmd_sweep` | mesh convergence, thickness studies |
| `query_field(field, region)` | nodal arrays | "max stress in the fillet region" |

Two rules make this safe. **Read tools run freely; write tools return a
proposal, not an effect** — `set_parameter` and `solve` come back as action
cards the user clicks, which is what the `APPLY` chips in the panel already
are. And **every numeric claim must cite a tool result**; if the model wants a
number it has not got, it must call a tool or say it doesn't know.

## System prompt, in essence

> You are a structural analysis assistant embedded in an FE workbench. You have
> tools for reading the study, reading results, and proposing changes. Never
> state a stress, displacement or safety factor that did not come from a tool
> result in this conversation. When the user asks for a design change, propose
> it as an action rather than describing it. Peak stress on a voxel mesh is
> mesh-dependent — say so when asked about stress concentrations, and suggest
> the gmsh mesher.

That last line matters: the assistant should know the failure modes of its own
solver. It has the convergence history in context and should read it.

## Worked example

> **"Is this design safe?"**
> → `get_results()` → SF 4.23 from σ_vm 118.9 MPa against 503 MPa yield.
> → `get_study()` → voxel mesher.
> → reads convergence history: peak stress still climbing with refinement.
> → *"SF 4.23 on the current mesh, but peak stress hasn't converged — it rose
> 109 → 137 MPa over the last four refinements because the voxel mesh
> staircases the fillet. Re-run on the gmsh mesher before trusting the margin."*
> → proposes `[APPLY] Switch to gmsh mesher and re-solve`.

That answer is only possible because the tools are the real solver and the
convergence history is real data. It is also the answer a good engineer gives,
which is the bar.

## What I would not build

- A vector database. The context is a few kB of structured JSON.
- Fine-tuning. Tool use over a well-specified schema covers it.
- Autonomous re-solving. Solves cost seconds to minutes and change what the
  user is looking at; keep a human on the trigger.
