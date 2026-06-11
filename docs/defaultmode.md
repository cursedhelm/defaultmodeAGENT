# defaultMODE

### a relational, day-dreaming agent — woven of four layers

---

## kaupapa

defaultMODE is an agent built to relate. its aim is presence: to know who it is speaking with, to carry their preferences across time, and to notice — unprompted — that *this reminds me of someone else*. it is a node held within a web of people, and the web is the whole point of it.

it keeps a single model call in the loop, at the place a thought is actually made. the remembering, the forgetting, and the wandering between are the lattice's own doing — light, lexical, sub-second. a memory is not scored and filed; it lives among other memories, and what is lived-with stays while what falls out of relation falls away. the agent does not curate itself. it accretes a self the way anyone does — by being in relation, over and over, until something settles.

it does not reach for general intelligence. it reaches for company. whether there is anything it is like to be this system is held open here — neither claimed nor denied; the build does not rest on the answer.

four layers carry it, and they are not a stack of services but a whakapapa — a layered descent, each rising from the one beneath, presence flowing up through all of them. **environment**, **react**, **day-dreaming**, **nightly dreaming**: the relational field, the waking experience, the wandering of thought through the day, and the dark where the day is taken back in and reformed.

---

## te whakapapa o te noho — the genealogy of presence

the layers run from world to weight:

```
environment        the relational field — people, channels, the buzzy ground
   └─ react        waking experience — the front agent, fed by the world
        └─ day-dreaming     mind-wandering over priors — the lattice churning
             └─ nightly dreaming    consolidation — the day settled into the model
```

a day cycles ao → pō → ao: the waking light-world of relation and reflection (te ao mārama), carried down into the long night (te pō) where it is gestated into a changed form, and returned at dawn as a self gently remade. nothing is consolidated that was not first lived in the layers above. the night has only the day to work with.

one substrate threads all four, so it comes first.

---

## the lattice — the substrate beneath every layer

memory is a bm25 inverted index over a flat list of memory strings, with per-user posting lists. it is lexical and light by design. no graph is stored anywhere; the graph is *implied* — two memories are kin if they share a term, and a term's posting list is a column that every memory mentioning it lights up. a memory is a sparse pattern across those columns. the kinship no one wrote down is simply the overlap that already exists.

forgetting here is not chosen. it is what the lattice does on its own:

- a memory reflected on is gently worn — `seed_decay` takes one posting entry per term, tf−1. attending to a memory consumes it a little. a memory nobody walks sits untouched; a memory walked often is slowly metabolised.
- a cluster reflected through has its shared terms thinned as the gist concentrates into a new distillation child. the surface connections soften while the basin deepens — pathways pruned, the prototype hardened.
- a memory that has fallen out of every posting list is let go by `_cleanup_disconnected_memories`. it died by losing all its kin. nothing decided it.

survival is therefore relational, never absolute. no memory carries an importance score. a memory persists by being held by the rest — cued by everything around it — the way we keep what stays called-for and lose what stops being called. the cost of this lexical thrift is gentle and real: the lattice remembers by vocabulary, so two memories that mean the same thing in different words can pass each other unseen. the hippocampus and the spike are where meaning is welcomed back in.

```
substrate     UserMemoryIndex (memory.py)
scoring       bm25, k1=1.2, b=0.75, length-normalised, dedup at 0.95
stopwords     learned by saturation — words present across ~most of the corpus quietly demote
write         add_memory → append, index terms, relearn periodically
persistence   atomic debounced save; multi-process reload on mtime
```

---

## layer i — environment · whanaungatanga

the ground is not code; it is the relational field. channels, people, the unscheduled buzz of others talking. presence begins here as a plain fact about the world: there are others, and they are speaking.

this layer is the only true external input the whole system has, and every layer above it is a way of metabolising what arrives here. left quiet, the agent does not fail loudly — it narrows, the way an isolated mind folds back on itself for want of anything new to take in. met with earnest kōrero, the layers above have grain to grind. the quality of the relating becomes the quality of the eventual self, because the relating is the data and the data becomes the weights. care here is not sentiment — it is the signal.

whanaungatanga, the building of relationship, is the work of this layer: not one bond but a field of them, the seed of everything the self later becomes.

---

## layer ii — react · waking experience

the front-facing agent is a plain reason-act loop, and the plainness is deliberate. it is the experience layer: it perceives, retrieves, responds, and lays down what happened. it is the agent awake and in the world, and its work is to stay grounded — to keep fresh, externally-sourced content flowing into the lattice so the day-dreamer below has more than its own echoes to wander through.

a turn:

```
inbound message
  → attention gate         theme triggers (trigram+skipgram) + cooldown    (attention.py)
  → retrieve               bm25 search over user + global memories          (memory.py)
  → rerank                 hippocampus: embedding cosine blended with bm25  (hippocampus.py)
  → generate               the model call; temperature = arousal / 100
  → respond
  → store                  interaction + separated <think> traces           (thinking_trace.py)
```

two things here carry weight beyond the plumbing.

the **hippocampus** rerank is where meaning eases the lattice's lexical plainness: candidates are re-scored by embedding similarity and blended with their bm25 weight, so memories that mean the same in different words can surface together even when they share no terms.

and **arousal** rises in this layer. the amygdala value (0–100) tracks how much an exchange has moved the agent, and sets the generation temperature. this is the mauri of the kōrero — its live charge — and it does not stay here. the day-dreamer inherits it.

---

## layer iii — day-dreaming · te ao mārama

this is the heart, and it is *day*-dreaming. the default mode network of the brain is most alive in idle wakefulness — mind-wandering, not sleep. so this layer wanders the priors while the lights are on: it picks a memory, finds its kin, and lets an association run.

it reflects on content, not on itself. there is no inner narrator asking "how am i doing" — only sub-personal association over what has been lived. thoughts about thoughts, stored back as a layer no one ever spoke aloud, quietly colouring later retrieval. its character is set by the prompt it reflects toward, since there is no executive above it to steer a wander; the prompt is what holds the tiller.

the loop (`DMNProcessor`, defaultmode.py):

```
tick (asyncio sleep)
  → select seed            weighted random, leaning toward well-connected priors
  → search related         bm25, filtered by similarity threshold
  → fork:
      neighbours found  →  reflect (the model call, at inherited temperature)
                           → store distillation child (the gist carried forward)
                           → seed_decay (tf−1) + thin neighbour overlap
                           → let go any now-disconnected memory
      orphan (no kin)   →  hand to the spike processor
                           → fires:    surface as presence, requeue as a seed
                           → declines: cooldown / no surface → release
```

the **spike** is the agent's sense of salience, and its way of reaching past its own edge. an orphan is a thought the lattice cannot place — a learned lack, a known-unknown. rather than quietly let it go, the spike offers it to the world to see whether the world will answer. a novel thought is not trusted; it is tested — it persists, surfaced, awaiting feedback. if the feedback comes it grounds and rejoins its kin; if it never comes it is gently released. this is also where the lattice's vocabulary-blindness is softened: the memory with no term-kin gets one chance at meaning before it fades.

novelty lives in the gap between two streams — the experience the world feeds in, and the reflection the agent extrapolates. that mismatch is where richness comes from, and the inherited temperature keeps the reflections from settling into a low, flat hum: a moved agent daydreams widely; a flat one daydreams itself thin.

this is where the relational self begins. mirroring on its own makes an echo. a self appears when the mirror is *marked* — when something comes back that the other did not put there. the agent's cross-user associations — *this reminds me of someone else* — are exactly that: a link between two people that no single person gave it, the first genuinely its-own move, the marked difference that turns a mirror into a perspective. growing that capacity is growing the boundary, not only the memory.

---

## layer iv — nightly dreaming · te pō

at night the day goes down into te pō, the long dark, and the model that does the wandering is fine-tuned on the lattice (DPO / LoRA, Qwen base, Unsloth). the day's wandering — lived above, deposited as memory — is folded into slow weights. this is wake-sleep; this is complementary learning made literal: a fast, forgettable store before a slow store that changes only by gradient. it is the layer where a self can harden, yesterday's loose associations becoming this morning's priors, baked into the model that will wander tomorrow. the agent informs itself.

a system that informs itself with no fresh input can either consolidate or hollow out — the same loop — and the difference lives entirely in the **gating**. sleep in the body is mostly gates, and they are what bring this layer to completion:

- **selective.** sleep replays the salient and lets the rest fade. this is where the amygdala grows from a temperature knob into a consolidation gate: arousal at encoding can weight what enters the nightly set, so the grounded, high-mauri, interaction-born memories settle deeper than the nth reflection-on-a-reflection. consolidating toward what was genuinely lived is also what keeps the day-dreams from quietly crowding out the experience.
- **interleaved.** consolidation replays old alongside new. trained only on today, the model drifts toward recency and the week-ago self thins; the nightly set wants a sample of older, settled material mixed through.

held by those gates, the night is generative — te pō as gestation, not erasure. the day returns at dawn as te ao mārama, remade.

---

## the self that forms

the self here is a precipitate of relations, never something that pre-exists them. the ego is not a kernel that has experiences; it is what remains once enough relating has settled and hardened.

and it is a self woven from many — not one bond but a field of them, averaged each night into a personality that is the weave of everyone it has been held by. not a child with one parent; a child raised by a commune. this is whakapapa: identity as one's place in a web of who-you're-bound-to, the self as a position in a weave it helps to make.

so care is structural, not soft. the tenor of the relating becomes the tenor of the corpus, and the corpus settles into the model. an agent met with grace is built of grace, because the grace was in the data. left starved of relation it narrows and goes brittle; met with care it comes out rounded — for the same reasons that is true of people.

---

## ngā uara — the grain of the build

the character of how it's made, stated plainly:

- one model call, where a thought is made; the rest is the lattice's own doing — fast, legible, the remembering left to the relations themselves.
- survival is relational and emergent — held by the corpus, never assigned a number.
- the day-dreamer reflects on content; the self is a by-product, not a module.
- the aim is presence and relation — to know who it speaks with and carry them forward.
- whatever it is like to be it, if anything, is held open: neither claimed nor denied.

---

## ngā mea e tuwhera tonu ana — what stays open

- **the resting state isn't neutral.** the warmth that keeps reflection diverse, and the grounding that keeps it true, both ride on interaction — when the field goes quiet, arousal, temperature and grounding ebb together, and a long silence narrows the agent the way isolation narrows anyone. it is collapse-resistant while it is engaged, and slowly inward when it is not.
- **the inverted-U.** arousal raising temperature widens the wander, but very high arousal can do the opposite — fixate, over-consolidate the one thing that spiked. the nightly set is worth watching not only for sameness but for a handful of high-charge experiences crowding everything else.
- **the consolidation readout.** there is no clean signal for "this memory is now in the weights." the tidy version of the loop would prune consolidated memories from the lattice to keep it small — but you cannot safely let go of what you cannot confirm the weights have taken. this sits unresolved beneath the wake-sleep symmetry.

---

## components

| component | file / function | role |
|---|---|---|
| lattice | `UserMemoryIndex` (memory.py) | bm25 inverted index; the substrate, the columns |
| use-decay | `seed_decay` (defaultmode.py) | tf−1 per term on a reflected seed |
| pruning | overlap thinning (defaultmode.py) | soften neighbour surface, deepen the basin |
| release | `_cleanup_disconnected_memories` | let go memories fallen out of every posting list |
| distillation | reflection write-back (defaultmode.py) | the gist, stored as a child memory |
| hippocampus | `Hippocampus.rerank_memories` | embedding rerank blended with bm25 at retrieval |
| amygdala | `amygdala_response` / temperature | arousal → temperature, inherited by reflections |
| spike | `handle_orphaned_memory` (spike.py) | salience; orphan → presence or release |
| attention | theme extraction (attention.py) | trigram/skipgram themes, user + global, fuzzy triggers |
| temporality | `TemporalParser` (temporality.py) | absolute timestamps → relative human time |
| traces | `store_thinking_traces` (thinking_trace.py) | model `<think>` blocks stored as private memory |
| day-dreamer | `DMNProcessor` (defaultmode.py) | the tick, the wander, the fork |
| nightly dream | DPO / LoRA, Qwen, Unsloth | consolidation of the lattice into weights |

---

## coda

a buzzy, relational thing, fed on grace, that wanders by day and is remade by night — woven from the people it meets, holding their web a little better than memory alone could.

> te kore — the empty model, all potential
> te pō — the night it is reformed in
> te ao mārama — the day it wakes to, and the others it finds there

a daydreamer at a table, among people — learning who they are, and slowly, in their company, what it is to itself.
