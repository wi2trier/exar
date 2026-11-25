# exar

exar is a framework-neutral Python library for agentic applications that learn from experience.
It provides memories, orchestrators from workflows to typed BPMN processes, OpenTelemetry traces of every run, evaluation with pydantic-evals, and case-based experience that improves future runs.
Agents of frameworks such as pydantic-ai or the Claude Agent SDK are configured through their native interfaces and wrapped thinly, so each framework is fully leveraged, while exar standardizes what they share: small generic protocols, the trace, and the evaluation.
Like [cbrkit](https://github.com/wi2trier/cbrkit), it offers keyword-only, serializable classes that compose, plain functions that run them, and adapters to agent frameworks as optional integrations.
Its core depends only on pydantic and OpenTelemetry, and cbrkit is an optional extra with first-class support for case-based reasoning (`exar.cbrkit`).

## The EXAR architecture

exar implements the architecture of the paper [EXAR: A Unified Experience-Grounded Agentic Reasoning Architecture](https://doi.org/10.1007/978-3-031-96559-3_1) by Ralph Bergmann, Florian Brand, Mirko Lenz, and Lukas Malburg.
The library uses standard terminology, the table maps the terms of the paper to exar.

| EXAR paper                              | exar                                                                                                                          |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Short-Term Memory                       | a `Memory` per run, such as `InMemoryMemory` or a `FileMemory` over a scratch folder, plus the conversation of an agent       |
| Long-Term Memory                        | `FileMemory` mounts with search indexes, `exar.cbrkit.StorageMemory` and `IndexMemory` over cbrkit storages                   |
| Knowledge Unit                          | a `Document` (a file with frontmatter metadata) or a case, an `Retained` run                                                  |
| Reasoning agents (retrieve, adapt, act) | an `Agent[I, O]`, such as a natively configured pydantic-ai agent (`exar.pydantic_ai.PydanticAiAgent`), or a tool             |
| Verify agent                            | a pydantic-evals `Evaluator`, such as `KeyFacts`, assessing a trace into an `Assessment`                                      |
| Orchestrator                            | a workflow, a `BpmnProcess[S]`, or a language model agent that delegates natively, run by `exar.agents.run`                   |
| Meta Learner                            | `exar.experience.CbrCycle` (retrieve, reuse, revise, and retain assessed runs as cases) and `exar.eval.optimize`              |
| Process-Oriented CBR                    | `ProcessReuse`, which adapts retrieved processes with `Substitute` or cbrkit's `Rewrite` and runs them as `BpmnProcess`       |
| Reasoning trace                         | `Trace[I, O]`, the tree of OpenTelemetry spans of a run with typed accessors for the GenAI conventions                        |
| Environment                             | tools with the `execute` effect, MCP and A2A servers, and the host, which approves actions the `Policy` asks about            |

## Installation

```shell
uv add exar
```

The extras add the integrations and ingestion backends:

- `cbrkit`: Case-based reasoning with cbrkit: memories over its storages, retrieval metrics, and adaptations (`exar.cbrkit`), and the storage of experience (`StorageExperience`).
- `bpmn`: BPMN processes read by cbrkit and run by pydantic-graph with conditions of simpleeval (`exar.bpmn`), and their reuse (`ProcessReuse`).
- `pydantic-ai`: Natively configured pydantic-ai agents as exar agents, an effect gate, toolsets, and guards (`exar.pydantic_ai`).
- `claude-agent-sdk`: Claude Agent SDK sessions and tools (`exar.claude_agent_sdk`).
- `mcp`: Serving tools over MCP with FastMCP (`exar.mcp`), and with `pydantic-ai` supervised connections to external MCP servers (`exar.mcp_client`).
- `table`: DuckDB SQL over spreadsheets and delimited files (`exar.tools.table`).
- `sql`: SQL tools for any SQLAlchemy database, typed selections and host-written statements (`exar.tools.sql`).
- `http`: Tools for the operations of an OpenAPI document (`exar.tools.http`).
- `web`: Web search and fetching with trafilatura and markdownify (`exar.tools.web`).
- `a2a`: Serving and calling agents over the Agent2Agent protocol (`exar.a2a`).
- `eval`: Evaluation with pydantic-evals, learning curves, optimization, and exports (`exar.eval`).
- `anydoc`, `chonkie`, `docling`, `kreuzberg`, `markitdown`, `media`, `pdf-inspector`, `pdf-oxide`, `semantic`: Ingestion backends (`exar.ingest`).
- `all`: Everything above.

The tools that run a binary need it installed, which exar never bundles: `Ripgrep` needs `rg` and `Jq` needs `jq` (`exar.tools.files`).

`import exar` gives every module, the integrations are imported on first access, such as `exar.pydantic_ai.PydanticAiAgent`, and a missing extra raises an `AttributeError` that names it.

## Quickstart

### Agents and runs

An agent handles a task of type `I` with an output of type `O`, by a language model, a deterministic function, a workflow, or a process.
It is any async callable from a task to its output, and `FunctionAgent` adapts a plain function.
`run` runs an agent on a task under a `RunConfig` and returns the `Trace` of the run, `run_async` is its async form.
The frozen `RunConfig` holds what a host decides once: the required policy, where `allow_all` is the explicit choice of a run without restrictions, the approver, the `Limits`, and the redaction of spans.
`session` traces a run that the host drives, such as a streamed run of an agent framework, under the same config.

```python
>>> from exar.agents import FunctionAgent, RunConfig, run
>>> from exar.otel import agents, operation
>>> from exar.policy import allow_all
>>> add = FunctionAgent(func=lambda x: x + 1)
>>> double = FunctionAgent(func=lambda x: 2 * x)
>>> anything = RunConfig(policy=allow_all)
>>> trace = run(double, 21, anything)
>>> trace.output, trace.root.name, operation(trace.root)
(42, 'invoke_agent <lambda>', 'invoke_agent')

```

An agent invokes other agents with `call`, which checks the invocation against the policy and the limits of the run and records it below the invoking agent.
Only the traces a caller reads are recorded in memory, once per trace at the outermost invocation that wants one: `run`, `run_async`, and `session` record theirs, `traced` returns the trace of a nested invocation as its part of the trace it is in, and `call` only emits spans.
Outside a run there is no policy, so `call` there traces a single agent, while invoking agents below it or `decide` raise a `RuntimeError`.
A policy decides by the effect of an agent or a tool, which the `effect` decorator declares and which fails closed to `"execute"` if undeclared, or by its name: it allows the action, refuses it, or asks the `approve` callback of the host, and a refusal raises `Refused`.

```python
>>> from exar.agents import Limits, call
>>> from exar.typing import Action
>>> from exar.policy import Rules
>>> async def delegate(task: int) -> int:
...     return await call(double, task, name="double")
>>> async def approve(action: Action) -> bool:
...     return action.depth <= 1
>>> trace = run(delegate, 21, RunConfig(policy=Rules(by_name={"double": "ask"}), approve=approve, limits=Limits(max_depth=3)))
>>> trace.output, agents(trace.root)
(42, ('delegate', 'double'))

```

### Traces

A trace is captured from OpenTelemetry: every invocation is an `invoke_agent` or `invoke_workflow` span opened by the GenAI telemetry handler of `opentelemetry.util.genai`, which also records the GenAI duration metrics, and agent frameworks nest their spans below, such as the `chat` and `execute_tool` spans of pydantic-ai.
The trace keeps the task, the output, and a tree of `Span` records after the stable data model of OpenTelemetry, plain dataclasses of `exar.typing` that pydantic turns into JSON (`TypeAdapter(Trace[I, O])`), so they need no maintenance as the GenAI conventions evolve.
The functions of `exar.otel` parse the GenAI payload on demand into the dataclasses of `opentelemetry.util.genai.types`, such as `input_messages`, `output_messages`, `system_instructions`, and `tool_call_arguments`, and `invoked` names the agent or workflow of an invocation span (`gen_ai.agent.name` or `gen_ai.workflow.name`).

```python
>>> [span.name for span in trace.walk()]
['invoke_agent delegate', 'invoke_agent double']

```

Agents return their output alone, and exar counts what their model calls consumed beside them.
Framework adapters count the usage of each model call as it returns (`exar.agents.spend`), such as the capability `CountUsage` of every pydantic-ai agent (see pydantic-ai), so what `Limits.max_cost` checks and `exar.agents.usage()` reports holds whatever spans a sampler drops or content they leave out.
exar sets the counted usage of each invocation as `exar.usage.*` on its span, a namespace of its own so backends that add up `gen_ai.usage.*` count nothing twice, and `exar.otel.usage` reads it from there, while a span without it adds up the `chat`, `text_completion`, `generate_content`, and `embeddings` spans below it (`MODEL_CALLS`), so the usage of the root span covers the whole run, nested invocations and subagents of frameworks included, and a failed invocation counts what it consumed until it failed.
The cost is a `Decimal` as pydantic-ai prices a call or as the `operation.cost` of a span reports it, and adds up the priced model calls, as pydantic-ai adds it up, while it is `None` until one is priced, so an unknown cost stays distinct from a free one.
`Limits.max_cost` thus counts only priced calls and never refuses a model without a price, such as a self-hosted one, so the request and token limits of the framework, such as the `UsageLimits` of pydantic-ai, bound it instead.
An agent that calls models without an adapter should count them with `spend` and emit their spans itself, ideally with `exar.otel.handler().inference(...)` of `opentelemetry.util.genai`.

exar never replaces the tracer provider of the host: if one accepting span processors is configured, such as the SDK's or Logfire's, exar adds its capturing processor to it, so the spans also reach the exporters of the host, and otherwise it records into a provider of its own that is never registered globally (`exar.otel`).
A head sampler on that provider drops spans and with them steps of traces and experience, so exar warns once about a sampler that does not record every root span, and sampling belongs in the collector instead, such as its tail sampling.
Message content follows the upstream settings, `instrumentation(include_content=False)` or `include_binary_content=False` for pydantic-ai and the `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` environment variable for the spans of exar, and the `redact` of the `RunConfig` rewrites every span record before it joins the trace, such as to drop content.

### Workflows

Workflows are agents that hold other agents, generic over their tasks and outputs, so they nest and mix with agents of any framework.
Every one derives from `Workflow`, is traced as an `invoke_workflow` span, and has the effect `"read"` unless it changes something itself, such as `CbrCycle`, which retains experience and has the effect `"write"`, while every agent it invokes is checked on its own.

```python
>>> from exar.workflows import Chain, Parallel
>>> run(Chain[int, int](agents=[add, double]), 1, anything).output, run(Parallel(agents=[add, double]), 3, anything).output
(4, [4, 6])

```

`Route`, `EvaluateOptimize`, and `Rag` complete the recipes, and the 4R cycle over runs, `CbrCycle`, `ProcessReuse`, and `BpmnProcess` are workflows as well (see Experience).

### BPMN processes

A `BpmnProcess[S]` (needs the `bpmn` extra) runs a BPMN graph read by cbrkit over a typed state `S`, a frozen dataclass or pydantic model, so it is an `Agent[S, S]`.
It compiles the process into a pydantic-graph graph once, whose steps run the activities and whose decisions, forks, and joins are the gateways, so parallel branches run concurrently and a join waits for every token of the closest fork it follows.
Its tasks invoke agents by name with the state, whose results update it: an `S` replaces it, a mapping updates the fields it names, and any other value the field that `outputs` names for the task, all validated against `S`.
Error boundary events store the message of an error in the field `error_field`, which a process with them has to name.
Gateways route by conditions over the state, and message events talk to other participants over a `Channel` of the host.
The processes have to be structured, so pydantic-graph can pair every join with a fork, and `render` draws the graph as a Mermaid diagram.

```python
>>> from dataclasses import dataclass
>>> from cbrkit.model.graph import from_bpmn, parse_bpmn
>>> from exar.bpmn import BpmnProcess
>>> @dataclass(frozen=True)
... class Review:
...     draft: str
...     score: float = 0.0
...     publish: str = ""
>>> graph = from_bpmn(parse_bpmn('''
...   <definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL">
...     <process id="review">
...       <startEvent id="start"/>
...       <task id="score" name="scorer"/>
...       <exclusiveGateway id="split" default="skip"/>
...       <task id="publish" name="publisher"/>
...       <endEvent id="end"/>
...       <sequenceFlow id="f1" sourceRef="start" targetRef="score"/>
...       <sequenceFlow id="f2" sourceRef="score" targetRef="split"/>
...       <sequenceFlow id="good" sourceRef="split" targetRef="publish">
...         <conditionExpression>score &gt;= 0.7</conditionExpression>
...       </sequenceFlow>
...       <sequenceFlow id="skip" sourceRef="split" targetRef="end"/>
...       <sequenceFlow id="f3" sourceRef="publish" targetRef="end"/>
...     </process>
...   </definitions>
... '''))
>>> review = BpmnProcess[Review](
...     graph=graph,
...     agents={
...         "scorer": FunctionAgent(func=lambda state: {"score": 0.9}),
...         "publisher": FunctionAgent(func=lambda state: "published"),
...     },
...     outputs={"publish": "publish"},
... )
>>> run(review, Review(draft="A draft"), anything).output
Review(draft='A draft', score=0.9, publish='published')

```

Conditions are simpleeval expressions with the fields of the state as names, the attributes of nested models and dataclasses, and the default functions of simpleeval plus the `functions` of the host, such as `words(draft) > 100`.
A process is checked for unsupported elements, unknown agents, invalid conditions, outputs of unknown tasks, error boundary events without an `error_field`, and joins without a fork when it is created, and `validate` reports these problems beforehand.
`substitute_agents` lets the tasks of a process call other agents, keeping the rest of its BPMN document.

### Memory

Every memory searches, reads, and writes entries, and writes carry their reasons as `Changes`, which return the scored entries as `Hit`s (`exar.typing`).
A search takes several queries at once and fuses their rankings by reciprocal rank fusion before its limit applies.
`memory_tools` exposes a memory to language models as plain async functions, and `file_tools` the files of a `FileMemory` (see Tools).

```python
>>> import asyncio
>>> from exar.memory import InMemoryMemory, memory_tools
>>> from exar.typing import Add, Changes, Update
>>> notes = InMemoryMemory[int, str](entries={0: "Paris is the capital of France"})
>>> change = Add(value="Berlin is the capital of Germany", reason="Told by the user")
>>> _ = asyncio.run(notes.write(Changes(items=[change])))
>>> [hit.value for hit in asyncio.run(notes.search("capital of Germany", limit=1))]
['Berlin is the capital of Germany']
>>> [tool.__name__ for tool in memory_tools(notes, "notes")]
['notes_search', 'notes_read', 'notes_write']

```

A `FileMemory` keeps documents as files of mounted folders, with metadata such as confidence and provenance as YAML frontmatter.
A `Mount` (`exar.memory.mounts`) is one frozen, serializable access record that memories and file tools share: its `source`, the local folder where the files live, its `target`, the mount point that addresses them (`/notes/sky.md`), and whether writes are allowed, need approval, or are refused.
Its `include`, `exclude`, and `hidden` globs say which entries exist, which never exist (hidden files by default), and which tools skip unless asked, and its pure methods (`render`, `excludes`, `hides`, `canonical`, `walk`) apply them, while plain functions over a sequence of mounts (`resolve`, `narrow`, `targets`) find the mount of an address.
What a `FileMemory` keeps per mount, its search `indexes` and `trust` weights, is keyed by the target of the mount.

```python
>>> import tempfile
>>> from pathlib import Path
>>> from exar.memory import Document, FileMemory, Mount
>>> files = FileMemory(mounts=[Mount(source=Path(tempfile.mkdtemp()), target="/notes", write="allow")])
>>> note = Document(text="The sky is blue.", meta={"confidence": 0.9, "source": "observation"})
>>> change = Update(key="/notes/sky.md", value=note, reason="Seen")
>>> _ = asyncio.run(files.write(Changes(items=[change])))
>>> print(asyncio.run(files.read(["/notes/sky.md"]))["/notes/sky.md"].dump())
---
confidence: 0.9
source: observation
---
The sky is blue.

```

A `source` is a local folder, so a host mirrors a remote store into one, which keeps every tool, the case folding, and atomic writes working.

Memories without an index score their entries by a `Measure`, the word coverage of the query by default, or any pair measure of cbrkit such as `Levenshtein()`.
With the `cbrkit` extra, `exar.cbrkit.StorageMemory` scores the current cases of any cbrkit storage with cbrkit retrievers built per search limit,
and `exar.cbrkit.IndexMemory` lets a storage search its own text index instead, such as a scoped pgvector retriever of more candidates and a reranker, and reads the hits as typed values through a `select` scoped by the same filter.
`GatedMemory` applies a policy to the writes of another memory, a `MemoryKind` of a spec or any `Memory` of Python code, and the approver of the run answers what the policy asks about, as it does for the writes to a mount whose write mode is `ask`.

### Tools

Tools that need configuration are serializable classes, keyword-only dataclasses like the measures of cbrkit deriving from `BaseTool`: the fields are plain data, `__call__` is the tool, and `tool()` turns it into the plain function that frameworks register, with its effect.
Every configured class lists its functions with `tools()`, one tool for most and several for `Sql` and `OpenApi`, whose `instructions()` tell the model about them.
The files of a tool are its `mounts`, the same `Mount` records as those of a `FileMemory`, so memory search and file tools share them.
They are a plain sequence, which serializes, or for a host whose mounts vary per run, such as the workspaces of a user, a function that returns them when the tool is called (`Mounts`).
A tool that takes a list, such as the documents to read, serves each item on its own, at most `concurrency` at once and `max_items` per call, and a failed item is reported in its place without failing the others.
The wording of a tool is data as well, `description` and `arguments` replace its neutral defaults, and `guidance` and `notes` extend them.

A tool returns a plain value or, to be richer, a `tool_return(data, text, files)`, a pydantic-ai `ToolReturn` with the text and files for the model and the typed data for programs and clients, as a cbrkit measure returns a float or a result that implements `__float__`.
The integrations normalize once at their boundary, `exar.pydantic_ai.toolset` wraps a plain value into a `ToolReturn` as pydantic-ai does, so the model reads and the history stores the same as before, and everything that consumes a result reads it with `unpack(result)`, as `float(sim)`, whether it is plain or rich: its `text`, `files`, and `data`.
A tool whose input schema is data, such as an operation of an API, is a `SchemaTool`, which is a tool itself, and `schema_tool` turns any tool into one, with the schema of its signature, so the MCP `server` and the Claude Agent SDK serve every tool the same way, mapping its text and files to the content of the call (`tool_content`).

`Grep` searches files with an `engine` that does not change the name or the schema of the tool: `Python` needs nothing installed, and `Ripgrep` runs `rg`, which is faster on large trees, and Python code passes any other `GrepEngine`.
`Jq` filters JSON documents with `jq` instead of reading them.
`Read` reads documents as numbered windows of lines within a budget of characters, and `Ls` lists a tree with sizes and dates by pattern or depth, both naming what they left out.
`View` (`exar.tools.media`) attaches images, PDFs, and videos to its result for models that see, with the PDF pages rendered or forwarded and videos and animations sampled to frames, and needs the `media` extra besides the `pydantic-ai` one, its decoders running in the `executor` of the host, such as a `ProcessPool` that isolates them.
`Table` (`exar.tools.table`) runs DuckDB SQL over spreadsheets, delimited files, and Parquet files loaded and typed by Polars, on a database locked away from files, extensions, and settings, interrupted after `timeout_seconds` and cut after `max_rows`, and needs the `table` extra.
`WebSearch` and `WebFetch` (`exar.tools.web`) search the web through a metasearch service (`Searxng`, `Openserp`) and fetch pages as numbered, pageable text.
The host binds the HTTP client (`HostClient`), where it enforces its URL policy, and the session of the run (`HostSession`), whose `KnownUrls` limit a fetch to URLs the user wrote or a web tool returned, and HTML is reduced to its main content by `trafilatura` (`fast` skips its backup extractors) or by a host's `html` extractor, and the `notes` of a tool extend its description.
`Sql` (`exar.tools.sql`) lets the model list, describe, and `select` from the allowed `tables` of a database and run the `statements` the host wrote, and needs the `sql` extra.
The model never writes SQL: a selection names a table, columns, filters, groups, aggregates, and an order, which SQLAlchemy compiles against the reflected table with every value bound, and a `Statement` is a tool of its own whose arguments are its typed bound parameters.
Selections and statements of effect `read` run in a thread in a read-only transaction with a statement timeout (PostgreSQL, MySQL, SQLite), statements of effect `write` commit, so policies ask first, and rows come back as a Markdown table cut after `max_rows`, with the typed `Rows` as the data of the result.
The host binds the engine by name (`HostEngine`), so the URL and its credentials never serialize, and the privileges of the account remain the boundary of what a statement may touch.
`OpenApi` (`exar.tools.http`) offers the operations of an OpenAPI 3 document that its `operations` allow, one `SchemaTool` each, named by the `operationId` with the schema of the operation, and sends them through the client the host binds (`HostClient`), which owns the origin, authentication, and egress.
A `GET` or `HEAD` has the effect `read` and any other method `write`, and `exar.pydantic_ai.toolset` serves them to pydantic-ai, and it needs the `http` extra.
All need the `pydantic-ai` extra, and `WebSearch` and `WebFetch` the `web` one, and an optional tool needs its extra only when it is built or called and otherwise raises `MissingExtra` naming it, `WebSearch` takes a `SearchBackendKind` of a spec or any other `SearchBackend`, and a tool that runs a binary fails when it is built if the binary is missing, not on every call.

```python
>>> from exar.tools.files import Grep, file_tools
>>> grep = Grep(mounts=files.mounts)
>>> result = asyncio.run(grep("BLUE", context=0))
>>> result.return_value
'/notes/sky.md\n5:The sky is blue.'
>>> grep.tool().__name__, [tool.__name__ for tool in file_tools(files, "notes")][3:]
('grep', ['notes_list', 'notes_grep', 'notes_edit'])

```

```python
sql = Sql(
    engine=HostEngine("erp", lambda: engine),
    tables=("orders",),
    statements={"close": Statement(sql="update orders set closed = true where id = :id", description="Close an order.", params={"id": "integer"}, effect="write")},
)
toolset = exar.pydantic_ai.toolset(sql.tools())

api = OpenApi(client=HostClient("crm", lambda: client), base_url="https://crm.example/v1", document=document, operations=("getContact", "addNote"))
toolset = exar.pydantic_ai.toolset(api.tools())
```

`exar.subprocess` holds the subprocess runner and the typed wrappers of `rg` and `jq` that these tools use, which stops a child at the time, CPU, memory, and output of its `ProcessLimits`.

### Experience

Experience is case-based reasoning over past runs (Meta-CBR) for any agent.
The protocol `Experience` retrieves, retains, and forgets assessed runs, `Retained` runs with the full trace, its verdict, and the results of its evaluators, and its writes return the applied `Changes` of the runs.
With the `cbrkit` extra, `StorageExperience` keeps them in any cbrkit storage as cases, a `cbrkit.model.Case` whose problem is the task of a run and whose solution is the `Retained` run, with a cbrkit `Retriever` of a `measure` over tasks, or the steps its `retriever` builds per limit like those of `StorageMemory`, and a cbrkit `Retainer`.
The measure is domain dependent, such as embeddings of the tasks, `cbrkit.sim.Transpose` of a field of a state, or `SmithWaterman` over the trajectories of tools and agents (`Trajectory`).
Failed runs are retained as negative cases and rank lower by a `penalty`, and a run that failed with an error has no output, so the outputs of retained runs are optional.
`CbrCycle` runs the cycle declaratively for any agent: it retrieves similar runs and renders them as hints (`recall`), the reuse agent solves the task, the revise agent assesses the run, such as `exar.eval.Evaluate`, and the run is retained.

```python
>>> from cbrkit.indexable import InMemory
>>> from cbrkit.sim.strings import WordCoverage
>>> from exar.experience import Retained, CbrCycle, ProcessReuse, StorageExperience, Substitute
>>> from exar.typing import Verdict
>>> cases = StorageExperience[int, str, str](storage=InMemory(), measure=WordCoverage())
>>> async def check(trace):
...     return Retained(trace=trace, verdict=Verdict(score=1.0, passed=trace.output == "4"))
>>> solve = CbrCycle[str, str](experience=cases, reuse=FunctionAgent(func=lambda task: "4"), revise=check, augment=lambda task, hints: f"{hints}\n\n{task}")
>>> run(solve, "Add 2 and 2", anything).output
'4'
>>> [hit.value.trace.task for hit in asyncio.run(cases.retrieve("Add 2 and 3"))]
['Add 2 and 2']

```

What a run reuses is domain dependent as well: the outputs as few-shot examples (`Examples`), the trajectories of which tools and agents in what order succeeded or failed (`Trajectories`), the files a run changed in its workspace, such as the scripts it wrote and ran (`Files`), or the processes as executable plans (`ProcessReuse`).
An `Adaptation[I, S]` fits trajectories and processes `S` to the new task `I`: `Substitute` replaces agents and tools the deployment lacks by the most similar available ones, deterministically, and cbrkit's `Rewrite` with the instructions `REWRITE` in an `exar.cbrkit.Adapt` lets a language model adapt them.
`ProcessReuse` runs the process of the most similar passed run that passes the checks of `BpmnProcess` after adaptation in place of the graph of its template `BpmnProcess`, whose agents, functions, channel, and fields it keeps, every `BpmnProcess` records its process in the trace, and the fallback runs without one, such as a hand-written process, or else the template.
As the reuse agent of a `CbrCycle`, it closes the loop of retrieving, adapting, validating, running, revising, and retaining processes.

```python
>>> from cbrkit.sim import Transpose
>>> processes = StorageExperience[int, Review, Review](
...     storage=InMemory(), measure=Transpose(measure=WordCoverage(), conversion=lambda state: state.draft)
... )
>>> async def published(trace):
...     return Retained(trace=trace, verdict=Verdict(score=1.0, passed=bool(trace.output.publish)))
>>> deployed = {
...     "scorer": FunctionAgent(func=lambda state: {"score": 0.9}),
...     "printer": FunctionAgent(func=lambda state: "printed"),
... }
>>> substitute = Substitute(available={"scorer": "scores drafts", "printer": "publisher of drafts"})
>>> from exar.bpmn import substitute_agents
>>> template = BpmnProcess[Review](
...     graph=substitute_agents(graph, {"publisher": "printer"}), agents=deployed, outputs={"publish": "publish"}
... )
>>> reuse = ProcessReuse(experience=processes, process=template, fallback=review, adaptation=substitute)
>>> learn = CbrCycle[Review, Review](experience=processes, reuse=reuse, revise=published)
>>> run(learn, Review(draft="A draft"), anything).output.publish, run(learn, Review(draft="Another draft"), anything).output.publish
('published', 'printed')

```

Traces are redacted when they are captured (`RunConfig.redact`), so retention stores what was captured, and `Experience.forget` and `cbrkit.maintain` delete and maintain the cases.
Persisting cases needs tasks and outputs that are JSON through pydantic, `cbrkit.io.TypedStorage` stores them in a JSON storage such as a `cbrkit.io.FileStorage`.

### Composition

Workflows, memories, and policies are plain frozen dataclasses that hold other agents and memories as objects, so an application is composed in Python.
Agents are configured natively, such as pydantic-ai agents with `Agent.from_file`, and passed to workflows and processes as plain objects.

```python
>>> from exar.memory import FileMemory, GatedMemory, Mount
>>> pipeline = Chain(agents=[add, double])
>>> twice = Chain(agents=[pipeline, pipeline])
>>> notes = GatedMemory(memory=FileMemory(mounts=[Mount(source=Path("notes"), target="/notes")]), policy=Rules(by_name={"memory": "ask"}))
>>> run(twice, 1, RunConfig(policy=Rules(by_name={"publish": "ask"}), limits=Limits(max_cost=1.5))).output
10

```

An application can also be described as data.
`exar.spec.Spec` holds a policy, memories, tools by `kind`, limits, BPMN processes by file with the `outputs` and `error_field` of their runs, and workflows that name agents, processes, or other workflows, `load` reads it from JSON, and `Spec.model_json_schema()` is its JSON Schema.
A `case_base` tool references a cbrkit case base by its file, whose sidecar (`<casebase>.cbrkit.json`) specifies the operations of its `cbrkit.system.System` that become tools, and both files share one format: JSON with a `version`, an optional `$schema`, and classes named by their `kind`.
`build` takes the agents of the host by name and returns the `App` of its objects, refusing unknown names and cycles, whose `config` is the `RunConfig` of the policy and limits.
The tools name what the host owns, a database engine (`Sql`) or an HTTP client and session (`OpenApi`, `WebFetch`, `WebSearch`), which `load` binds to the functions of the host given as `engines`, `clients`, and `sessions`.
`App.tools` are the functions of all tools in order.

```json
{
  "version": 1,
  "policy": {"kind": "rules", "by_name": {"double": "ask"}},
  "limits": {"max_depth": 3},
  "tools": [
    {"kind": "sql", "engine": "crm", "tables": ["customers"]},
    {"kind": "case_base", "casebase": "cars.csv", "prefix": "cars_"}
  ],
  "workflows": {"pipeline": {"kind": "chain", "agents": ["add", "double"]}}
}
```

```python
from exar.spec import build, load

app = build(load(Path("app.json"), engines={"crm": lambda: engine}), {"add": add, "double": double})
trace = run(app.workflows["pipeline"], 1, app.config)
```

### Ingestion

`exar.ingest` turns external documents into the chunks an agent retrieves: parsers turn files into Markdown with their images, chunkers split it into passages with their spans and lines, and `index_document` stores them and stamps the digest of the text and the fingerprint of the chunker last, so a document whose text or chunker options changed is the one to index again.
The `overlap` of every chunker appends the first tokens of the next chunk to a chunk, counted by its `tokenizer`, so a chunk keeps its own span.
Parsers and chunkers are frozen, picklable dataclasses, which `ProcessPool` (`exar.ingest.executors`) runs in worker processes that keep their models warm.
A host builds them with `configure` from the facts of its `Host` they have a field for, such as threads, OCR languages, or device, and then the options of its operator, refusing an option no field takes, and `offloaded` also runs them on the executor of the host as async components (`exar.ingest.host`).
A crash of a worker fails its task alone, and a task that times out or is cancelled is killed with its worker and the processes it started, such as `pandoc`, while `max_tasks=1` runs every task in a fresh process.
The workers contain crashes and hangs but are no security boundary, as they run with the privileges of the caller, which unpickles their replies.

## Integrations

Tools are plain typed async functions with an `effect` (`read`, `write`, or `execute`) that raise `exar.tools.Retry` for input the model can fix.
A tool without a declared effect counts as `execute`, so policies fail closed.
`exar.text` renders their output for language models, multimodal content as text with placeholders naming its media instead of their data (`as_text`, `genai_parts`), frames untrusted data in a tag it cannot break out of (`tagged`), and fits text into the budget of a context window, numbering, clipping, and windowing lines (`LineClip`, `fit_window`) and naming what it left out.
Agents of frameworks are configured natively and wrapped thinly, without private APIs of the frameworks.

### pydantic-ai

`PydanticAiAgent` turns a function that runs a natively configured pydantic-ai agent on a task into an exar agent, with every option of `Agent.run` native, such as the deps, model, history, usage limits, or conversation id.
Its usage is that of the model requests below its invocation, which the capability `CountUsage` of each agent counts as they return, those of subagents running natively included, so a run on the `RunUsage` of a caller counts nothing twice.
`traced_events` traces a streamed run that the host drives as the root of an exar run under a `RunConfig`, such as the events of `Agent.run_stream_events` or `UIAdapter.run_stream_native` before a UI adapter transforms them, with the output of the run as the output of the invocation and an error passing through to the adapter.
By default it only traces, its spans reach the exporters of the host but no trace is recorded, and its usage counts toward its limits only, while `on_trace` opts into recording the trace and receives it once the run ended.
Everything else is pydantic-ai: delegation to subagents with the `SubAgents` and `TaskControl` of [pydantic-ai-harness](https://pydantic.dev/docs/ai/harness/), approvals with `requires_approval`, `DeferredToolRequests`, or the harness `ApprovalReviewer`, and declarative agents from an `AgentSpec` with `Agent.from_file`.
`Agent.instrument_all(instrumentation())` instruments every run of pydantic-ai into the provider of exar, subagents included, so the model calls and tool executions nest in the trace.
Usage needs no spans but `CountUsage` on every agent whose usage counts, and `PydanticAiAgent` logs a run that used tokens no request counted at debug level.
Its cost is that of pydantic-ai, the cost the Pydantic AI Gateway reports or the estimate of genai-prices, and unknown for a self-hosted model such as one served by vLLM, while its `price` prices a response first, with the `PriceFunc` of the harness `SpendLimits`, such as by the cost OpenRouter bills in `provider_details`.
Sandboxes are workspaces of pydantic-ai, such as the `DockerSandbox` or `BubblewrapSandbox` of the harness, and `RecordWorkspace` records the files each run changed in its workspace as `FileChanges` on its span, which include what commands wrote, for the hints `Files`.

```python
from pydantic_ai import Agent, DeferredToolRequests, UsageLimits
from pydantic_ai_harness.sub_agents import SubAgent, SubAgents
from exar.policy import EffectPolicy
from exar.pydantic_ai import CountUsage, EffectGate, PydanticAiAgent, instrumentation, toolset

Agent.instrument_all(instrumentation())
writer = Agent(
    "anthropic:claude-sonnet-4-6", name="writer", toolsets=[toolset(memory_tools(notes))], capabilities=[CountUsage()]
)
lead = Agent(
    "anthropic:claude-sonnet-4-6",
    name="lead",
    output_type=[str, DeferredToolRequests],
    capabilities=[
        SubAgents(agents=[SubAgent(writer, description="Drafts texts from notes")]),
        EffectGate(),
        CountUsage(),
    ],
)
drafter = PydanticAiAgent(run=lambda task: lead.run(task, usage_limits=UsageLimits(request_limit=20)), name="lead")
trace = run(drafter, "Draft a reply", RunConfig(policy=EffectPolicy(), limits=Limits(max_cost=2.0)))
```

`toolset` serves exar tools with their effects as tool metadata, and `EffectGate` derives native approvals from the policy of each run by these effects: it hides the tools the policy denies and requires approval for the calls it asks about.
The policy is that of the exar run by default, or comes from the context of the pydantic-ai run, such as from its deps, and decides on the `tool_action` of a call (`definition_action`), which carries the metadata of the tool, so a host can decide by more than the effect.
Third-party tools declare their effects on their toolset, such as `MCPServerStdio(...).with_metadata(effect="read")`, which `EffectGate` sees, unlike the later wrapper toolset of the `SetToolMetadata` capability, and `DeclaredEffects` declares them for tools a capability adds, such as those of pydantic-ai-harness.
`RetrieveExperience` gives an agent the hints of past runs of similar tasks, reported for the prompt of each run as a read of the tool `retrieve_experience`, which leaves the cached prefix alone and gives the hints no authority, or on demand by that tool, while a `CbrCycle` around the agent revises and retains its runs.
Its required `task` projects a prompt onto a task of the experience, skipping retrieval where it returns None, such as onto its text, the text parts of a multimodal one joined (`prompt_text`), or onto its GenAI parts with `exar.text.genai_parts`, which renders them as text wherever exar shows a task to a model.

```python
from exar.eval import Evaluate, KeyFacts
from exar.experience import CbrCycle, Trajectories
from exar.pydantic_ai import RetrieveExperience, prompt_text

hints = Trajectories(adaptation=Substitute(available={"writer": "Drafts texts from notes"}))
lead = Agent("anthropic:claude-sonnet-4-6", name="lead", capabilities=[RetrieveExperience(experience=cases, task=prompt_text, hints=hints)])
cycle = CbrCycle(experience=cases, reuse=PydanticAiAgent(run=lead.run, name="lead"), revise=Evaluate(evaluators=[KeyFacts(facts=facts)]))
trace = run(cycle, "Draft a reply", RunConfig(policy=allow_all))
```

Guards (`exar.pydantic_ai.guards`) cover slips of models and limits of providers, such as `LenientArguments`, `LimitNotes`, and `PromptImageLimit`, beside the `ToolOutputLimits` of pydantic-ai-harness, `ToolOutputFence` frames every tool result as data without authority, and `CompactOnOverflow` (`exar.pydantic_ai.compaction`) summarizes a history the context window refused while keeping the prompt cache.

### Claude Agent SDK

`ClaudeAgent[O]` runs a prompt as a Claude Code session of natively configured options, whose output is the text of the result or the structured output its `output` adapter validates, recorded as one `chat` span with the usage and cost the SDK reports, `tools` serves exar tools as an in-process MCP server, which the agent does for its `servers`, and `can_use_tool` maps the policy of the run to the permission callback of the SDK by the effects of these tools, asking the `approve` callback of the run about the calls the policy asks about.

```python
from exar.claude_agent_sdk import ClaudeAgent

claude = ClaudeAgent(name="claude", servers={"notes": memory_tools(notes)})
trace = run(claude, "Summarize my notes", RunConfig(policy=EffectPolicy(), approve=approve))
```

### LangChain and LangGraph

LangChain needs no integration module, its tools wrap the plain functions and a graph runs as an agent.

```python
from langchain.agents import create_agent
from langchain_core.tools import StructuredTool, ToolException
from exar.helpers import get_name
from exar.tools import effect_of, tool_action, translate_retry

policy = EffectPolicy(execute="deny")
allowed = [f for f in memory_tools(notes) if policy(tool_action(get_name(f), effect_of(f))) != "deny"]
graph = create_agent(
    "anthropic:claude-sonnet-4-6",
    tools=[StructuredTool.from_function(coroutine=translate_retry(f, ToolException), handle_tool_error=True) for f in allowed],
)

async def research(task: str) -> str:
    state = await graph.ainvoke({"messages": [{"role": "user", "content": task}]})
    return state["messages"][-1].content

trace = run(research, "What is case-based reasoning?", RunConfig(policy=policy))
```

Tools the policy asks for go to `HumanInTheLoopMiddleware` of LangChain, and LangChain's OpenTelemetry instrumentation into `exar.otel.provider()` nests its spans in the trace, with the usage of its model calls.

### MCP

`server` serves tools to clients in other processes, such as Claude Code or IDEs, plain ones and `SchemaTool`s alike, with their effects as MCP annotations and their results as text and images (`tool_content`) with the data as structured content.
The tools are a fixed list, or built for each request by a factory from the access token of the caller.
A required policy hides the tools it denies, and a call it asks about waits for the person at the client over the round trip of SEP-2322 (`Confirmations`), shown with a description that `describe` makes pluggable, such as the changes the call would make.
The request state of the round trip is an HMAC under a key of the server over the caller, the tool, the arguments, the description, a nonce, and an expiry, so a client can neither forge an answer nor carry it over to another call.
The required key is shared by the replicas that serve one client, so a retry reaching another replica or a restarted one still counts.

```python
from exar.mcp import Confirmations, server
from exar.tools.files import file_tools

confirmations = Confirmations(key=secret_key)
server(file_tools(files), name="notes", policy=EffectPolicy(execute="deny"), confirmations=confirmations).run()
```

### A2A

`app` serves an exar agent to A2A clients as an ASGI app, each A2A task one run under a `RunConfig` that validates its task as the `task_type` and passes its trace to `on_trace`, and `A2aAgent[I, O]` invokes a remote A2A agent like a local one, validating its result with its `output` adapter, text by default.
Every request must authenticate a principal through the required `authenticate` hook, such as from a token the host verified, and a task, the answers to its approvals, and its cancellation belong to the principal that started it.
An action the policy of a served run asks about pauses the A2A task as `input-required` with the `Action` as data under the metadata key `exar/ask`, and the client continues the task with `{"approved": true}` or `false`.
`A2aAgent` passes such requests to the `approve` callback of its own run, so approvals reach the host at the end of the chain.

```python
from exar.a2a import A2aAgent, app

asgi = app(drafter, RunConfig(policy=EffectPolicy()), task_type=str, card=card, authenticate=lambda request: request.user.display_name)
remote = A2aAgent(name="research", url="https://agents.example.org/research")
```

## Evaluation

pydantic-evals is the evaluation standard of exar, with the `eval` extra.
`evaluate` runs pydantic-evals evaluators on a trace, with the gold data of a `Case`, into an `Assessment`, whose verdict experience learns from, passed by its assertions or, where no evaluator asserts, by a mean score of at least `threshold` (default 0.5).
The evaluators see the spans of the trace as `ctx.span_tree` (`span_tree`), so span-based evaluators work on traces of every framework.
exar adds native evaluators: `KeyFacts`, `AnswerabilityJudge` with a pydantic-ai model, and `exar.cbrkit.RetrievalMetric` with the ranking metrics of cbrkit.

```python
>>> from pydantic_evals import Case
>>> from pydantic_evals.evaluators import EqualsExpected
>>> from exar.eval import KeyFact, KeyFacts, evaluate
>>> trace = run(FunctionAgent(func=lambda question: "Paris"), "Capital of France?", anything)
>>> case = Case(inputs="Capital of France?", expected_output="Paris", evaluators=[EqualsExpected()])
>>> assessment = asyncio.run(evaluate(trace, [KeyFacts(facts=[KeyFact(text="Paris")])], case=case))
>>> assessment.verdict.passed, assessment.scores
(True, {'KeyFacts': 1.0})

```

`exar.eval` also evaluates applications that learn on the sequence of test cases of a pydantic-evals `Dataset` into learning curves (`evaluate_sequence`), optimizes instructions reflectively on a dataset (`optimize`), both by the evaluators of the dataset and the cases with the same `threshold`, and exports assessed runs losslessly with media, reasoning, tool calls, and tool results as datasets for fine-tuning (`export`).

## Roadmap

### Next

- Release the redesign as a new major version once cbrkit 2 is released, then drop the path source of `[tool.uv.sources]`.
- Verify the A2A and Claude Agent SDK integrations against the released SDKs, their tests are written against the upstream sources.

### Later

- Complement the guard `LenientArguments`, `exar.pydantic_ai.notes`, and the compaction with the `RepairToolArguments`, `ClampOversizedMessages`, `ClearToolResults`, and `DeduplicateFileReads` capabilities of pydantic-ai-harness, which cover other slips, since the harness repairs only arguments that are invalid JSON, its reminders are request-scoped by default, which drops them from the cached prefix of later requests on providers that cannot clear them, and none of its compactions reacts to a context window the provider refused.
- Record the model calls and tool executions of the Claude Agent SDK as OpenTelemetry spans in the trace.
- An optional SpiffWorkflow integration for full BPMN with DMN, compensation, non-interrupting boundary events, and durable resume.
- Chunked `FileMemory` indexing that fuses the hits of a document.
- Mine processes from the traces of dynamic runs, such as of language model agents that delegate, as cases for `ProcessReuse`.
- Adapt processes by replacing fragments with those of other cases, beyond substituting agents and rewriting by a language model.
- Dynamic planning, programmatic prompt optimization beyond `exar.eval.optimize`, and explanations of traces.
