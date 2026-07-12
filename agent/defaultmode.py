import asyncio
import random
from collections import defaultdict
import logging
from datetime import datetime
import re
from pydantic import BaseModel, Field
from chunker import truncate_middle, clean_response
from temporality import TemporalParser
from thinking_trace import separate_thinking_traces, store_thinking_traces
from fuzzywuzzy import fuzz
try:
    from tools.chronpression import chronomic_filter as _chronomic_filter
except ImportError:
    _chronomic_filter = None


class DMNPrompts(BaseModel):
    """Hardcoded prompt scaffolding for DMN thought generation.

    The thought_memory template is load-bearing: its prefix ("Reflections/
    Distillation on priors with @...") is how DMN thoughts identify themselves
    when they resurface in prompts, and spike.extract_memory_content splits on
    ':\\n' to strip it. The instruction prompts live in the bot's YAML
    (generate_dmn_thought / dmn_thought_generation).
    """
    connected_memories_header: str = Field(default="{count} Connected memories:\n\n")
    memory_entry: str = Field(default="{memory} [Weight: {score:.2f}]\n\n")
    empty_recall: str = Field(default="Hmmm... nothing comes to mind.\n")
    llm_label: str = Field(default="Reflections")
    chronpression_label: str = Field(default="Distillation")
    users_suffix: str = Field(default=" and {users}")
    thought_memory: str = Field(default="{label} on priors with @{name}{users} {timestamp}:\n{thought}")


PROMPTS = DMNPrompts()


class DMNProcessor:
    """
    Default Mode Network (DMN) processor that implements background thought generation
    through random memory walks and associative combination.
    """
    def __init__(self, memory_index, prompt_formats, system_prompts, runtime, dmn_config=None, mode="conservative", dmn_api_type=None, dmn_model=None):
        # Core components
        self.memory_index = memory_index
        self.prompt_formats = prompt_formats
        self.system_prompts = system_prompts
        self.runtime = runtime
        # Keep bot as an alias for backwards-compatibility with any external callers
        self.bot = runtime
        # Use runtime's logger
        self.logger = runtime.logger if hasattr(runtime, 'logger') else logging.getLogger('bot.default')
        # Load DMN configuration
        if dmn_config is None:
            from bot_config import config
            dmn_config = config.dmn
        # Operational settings
        self.tick_rate = dmn_config.tick_rate
        self.enabled = False
        self.task = None
        # Thought generation parameters
        self.temperature = dmn_config.temperature
        self.amygdala_response = 50  # Default intensity
        self.combination_threshold = dmn_config.combination_threshold
        # Memory decay settings
        self.decay_rate = dmn_config.decay_rate
        self.top_k = dmn_config.top_k
        # Retrived Memory Density Temperature Multiplier settings
        self.density_multiplier = dmn_config.density_multiplier
        # Fuzzy matching settings
        self.fuzzy_overlap_threshold = dmn_config.fuzzy_overlap_threshold
        self.fuzzy_search_threshold = dmn_config.fuzzy_search_threshold
        # Memory context compression settings
        self.max_memory_length = dmn_config.max_memory_length
        self.temporal_parser = TemporalParser()  # Add temporal parser instance
        # Chronomic distillation settings
        self.use_chronpression = dmn_config.use_chronpression
        self.chron_compression_max = dmn_config.chron_compression_max
        # Search similarity settings
        self.similarity_threshold = dmn_config.similarity_threshold
        # Neighbour thinning probability bounds
        self.thin_p_min = dmn_config.thin_p_min
        self.thin_p_max = dmn_config.thin_p_max
        # Store modes from config
        self.modes = dmn_config.modes
        # Set initial mode
        self.set_mode(mode)
        # DMN-specific API settings
        self.dmn_api_type = dmn_api_type
        self.dmn_model = dmn_model
        # Seeds queued for priority processing (e.g. post-spike orphans)
        self.pending_seeds: list = []

        self.logger.info(f"DMN Processor initialized with API: {dmn_api_type or 'default'}, Model: {dmn_model or 'default'}")

    async def start(self):
        """Start the DMN processing loop."""
        if not self.enabled:
            self.enabled = True
            self.task = asyncio.create_task(self._process_loop())
            self.logger.info("DMN processing loop started")

    async def stop(self):
        """Stop the DMN processing loop."""
        if self.enabled:
            self.enabled = False
            if self.task:
                self.task.cancel()
                try:
                    await self.task
                except asyncio.CancelledError:
                    pass
                self.task = None
            self.logger.info("DMN processing loop stopped")

    def set_amygdala_response(self, intensity: int):
        """Update amygdala arousal and temperature scaling."""
        self.amygdala_response = intensity
        self.temperature = intensity / 100.0
        self.logger.info(f"Updated amygdala arousal to {intensity} (temperature: {self.temperature})")

    async def _process_loop(self):
        """Main DMN processing loop."""
        while self.enabled:
            try:
                await self._generate_thought()
            except Exception as e:
                self.logger.error(f"Error in DMN thought generation: {str(e)}")
            await asyncio.sleep(self.tick_rate)

    def _select_random_memory(self):
        """Select a random memory based on contextual weights."""
        user_ids = list(self.memory_index.user_memories.keys())
        if not user_ids:
            return None
        # Use memory count directly as weights (not ranks)
        user_weights = []
        user_memory_counts = []
        for user_id in user_ids:
            memory_count = len(self.memory_index.user_memories[user_id])
            user_weights.append(memory_count)
            user_memory_counts.append((user_id, memory_count))
        # Sort by memory count for logging (highest first)
        user_memory_counts.sort(key=lambda x: x[1], reverse=True)
        # Log top users by memory count with names
        top_users = user_memory_counts[:5]  # Show top 5
        top_users_with_names = []
        for user_id, count in top_users:
            try:
                # Use sync cache lookup if available (Discord), else fall back to id string
                get_user = getattr(self.runtime, 'get_user', None)
                user = get_user(int(user_id)) if get_user else None
                user_name = user.name if user else f"Unknown({user_id})"
            except Exception:
                user_name = f"Unknown({user_id})"
            top_users_with_names.append((user_name, count))
        
        self.logger.info(f"User memory ranking - Top users: {top_users_with_names}")
        # Weighted random selection
        total_weight = sum(user_weights)
        if total_weight <= 0:
            selected_user_id = random.choice(user_ids)
        else:
            selection_point = random.uniform(0, total_weight)
            current_weight = 0
            selected_user_id = user_ids[0]  # fallback
            
            for user_id, weight in zip(user_ids, user_weights):
                current_weight += weight
                if current_weight >= selection_point:
                    selected_user_id = user_id
                    break
        user_memories = self.memory_index.user_memories[selected_user_id]
        if not user_memories:
            return None
        # Derive weights from TF totals in the inverted index (sum of posting list appearances per mid)
        tf_totals = defaultdict(int)
        for pl in self.memory_index.inverted_index.values():
            for mid in pl:
                tf_totals[mid] += 1
        # Build (mid, text, weight) triples for user's memories
        weighted_memories = [
            (memory_id, self.memory_index.memories[memory_id],
             max(1, tf_totals.get(memory_id, 1)))
            for memory_id in user_memories
            if self.memory_index.memories[memory_id] is not None
        ]
        if not weighted_memories:
            return None
        total_weight = sum(w for _, _, w in weighted_memories)
        if total_weight <= 0:
            return None
        # Random selection based on weights
        selection_point = random.uniform(0, total_weight)
        current_weight = 0
        for memory_id, memory, weight in weighted_memories:
            current_weight += weight
            if current_weight >= selection_point:
                return selected_user_id, memory_id, memory

        return None

    async def _generate_thought(self):
        """Generate new thought through memory combination and insight generation."""
        max_retries = 8
        from_pending = False
        for attempt in range(max_retries):
            if self.pending_seeds and attempt == 0:
                user_id, seed_mid, seed_memory = self.pending_seeds.pop(0)
                from_pending = True
                self.logger.info(f"dmn.pending_seed consumed user={user_id} mid={seed_mid} memory={seed_memory[:80]}")
            else:
                from_pending = False
                selection_result = self._select_random_memory()
                if not selection_result:
                    return
                user_id, seed_mid, seed_memory = selection_result
            
            try:
                user_name = await self.runtime.resolve_user(user_id)
            except Exception:
                user_name = "Unknown User"
            # Run memory search in executor to prevent blocking
            try:
                loop = asyncio.get_event_loop()
                related_memories = await loop.run_in_executor(
                    None,
                    lambda: self.memory_index.search(seed_memory, user_id=user_id, k=int(self.top_k))
                )
            except Exception:
                continue
            # Filter out the seed memory from results and apply weight threshold
            related_memories = [
                (memory, score) for memory, score in related_memories 
                if memory != seed_memory and score >= self.similarity_threshold
            ]
            # Log similarity threshold filtering results
            self.logger.info(f"After similarity threshold ({self.similarity_threshold}): {len(related_memories)} memories selected")
            # If we found any related memories, we can proceed
            if related_memories:
                break
            # No related memories - this is an orphan, delegate to spike immediately
            # If this seed came from pending_seeds it already had its spike chance — bail cleanly
            if from_pending:
                self.logger.info("dmn.pending_seed still orphaned after spike—dropping")
                self._cleanup_disconnected_memories()
                return
            self.logger.info(f"dmn.orphan detected—delegating to spike")
            if self.runtime.spike_processor:
                from spike import handle_orphaned_memory
                fired = await handle_orphaned_memory(self.runtime.spike_processor, seed_memory)
                if fired:
                    # Queue under bot's own user_id so the next search finds the spike
                    # interaction memory (stored under bot.user.id, not the original user)
                    bot_uid = self.runtime.agent_id
                    self.logger.info(f"spike.fired from dmn orphan—queuing under bot_uid={bot_uid} for dmn reprocessing")
                    entry = (bot_uid, seed_mid, seed_memory)
                    if entry not in self.pending_seeds:
                        self.pending_seeds.append(entry)
                    self._cleanup_disconnected_memories()
                    return
                else:
                    self.logger.info("spike.declined (no viable surface or cooldown)—retrying seed")
            # If spike didn't fire, continue retry loop for a new seed
            self.logger.info(f"Attempt {attempt + 1}: No related memories found, trying another seed memory")
            if attempt == max_retries - 1:
                self.logger.info("Max retries reached without finding related memories or spike target")
                self._cleanup_disconnected_memories()
                return

        # Log DMN process start
        self.logger.log({
            'event': 'dmn_process_start',
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'user_name': user_name,
            'seed_memory': seed_memory,
            'related_memories_count': len(related_memories)
        })

        # Build memory context using ALL related memories
        memory_context = PROMPTS.connected_memories_header.format(count=len(related_memories))
        if related_memories:
            for memory, score in sorted(related_memories, key=lambda x: x[1], reverse=True):
                # Convert any timestamp in the memory to temporal expression
                timestamp_pattern = r'\((\d{2}):(\d{2})\s*\[(\d{2}/\d{2}/\d{2})\]\)'
                
                parsed_memory = re.sub(timestamp_pattern, 
                    lambda m: f"({self.temporal_parser.get_temporal_expression(datetime.strptime(f'{m.group(1)}:{m.group(2)} {m.group(3)}', '%H:%M %d/%m/%y')).base_expression})", 
                    memory)
                memory_context += PROMPTS.memory_entry.format(memory=parsed_memory, score=score)
        else:
            memory_context += PROMPTS.empty_recall

        # Get high-similarity memories for term processing
        similar_memories = []
        for memory, score in related_memories:
            # Apply fuzzy matching to memory content
            content_ratio = fuzz.token_sort_ratio(seed_memory, memory)
            if content_ratio >= self.fuzzy_overlap_threshold or score >= self.combination_threshold:
                similar_memories.append((memory, max(score, content_ratio/100.0)))
                self.logger.info(f"Memory matched with fuzzy ratio: {content_ratio}%, semantic score: {score:.2f}")

        # Process overlapping terms if we have high-similarity memories
        if similar_memories:
            # Sort by combined score
            similar_memories.sort(key=lambda x: x[1], reverse=True)
            # Log memory processing
            self.logger.log({
                'event': 'dmn_memory_processing',
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'similar_memories_count': len(similar_memories),
                'combination_threshold': self.combination_threshold,
                'memory_context': memory_context
            })
            # Resolve mids once, before any await, under the lock to guard against reload races.
            # Use the carried seed_mid (fix 6) — only fall back to index() for neighbours.
            seed_memory_id = seed_mid
            score_by_mid = {}
            top_memories = [(seed_memory, seed_memory_id)]
            with self.memory_index._mut:
                for memory, s in similar_memories[1:]:
                    try:
                        mid = self.memory_index.memories.index(memory)
                    except ValueError:
                        continue
                    top_memories.append((memory, mid))
                    score_by_mid[mid] = float(s)
            # Find overlapping terms between seed and each result
            memory_terms_map = {}
            for memory, memory_id in top_memories:
                memory_terms = set(
                    term for term, ids in self.memory_index.inverted_index.items()
                    if memory_id in ids
                )
                memory_terms_map[memory_id] = memory_terms
            # Find terms that overlap between seed and each result
            seed_terms = memory_terms_map[seed_memory_id]
            overlapping_terms = set()
            fuzzy_matches = defaultdict(set)
            # First pass - exact matches
            for memory_id in memory_terms_map:
                if memory_id != seed_memory_id:
                    overlapping_terms.update(seed_terms & memory_terms_map[memory_id])
            # Second pass - fuzzy matches
            for seed_term in seed_terms:
                for memory_id in memory_terms_map:
                    if memory_id != seed_memory_id:
                        for term in memory_terms_map[memory_id]:
                            # Skip if terms are identical (prevent self-matches)
                            if seed_term.lower() == term.lower():
                                continue
                            ratio = fuzz.ratio(seed_term.lower(), term.lower())
                            if ratio >= self.fuzzy_search_threshold:  # Threshold for fuzzy matches
                                fuzzy_matches[seed_term].add(term)
                                overlapping_terms.add(term)

            if overlapping_terms or fuzzy_matches:
                self.logger.info(f"Found {len(overlapping_terms)} exact overlapping terms")
                self.logger.info(f"Found {len(fuzzy_matches)} fuzzy term matches")
                for seed_term, matches in fuzzy_matches.items():
                    self.logger.info(f"Fuzzy matches for '{seed_term}': {', '.join(matches)}")
                
                # Store the state for post-processing
                memory_update_state = {
                    'user_id': user_id,
                    'top_memories': top_memories,
                    'memory_terms_map': memory_terms_map,
                    'overlapping_terms': overlapping_terms,
                    'score_by_mid': score_by_mid,
                }

        # Replace timestamp generation with temporal expression
        current_time = datetime.now()
        temporal_expr = self.temporal_parser.get_temporal_expression(current_time)
        timestamp = temporal_expr.base_expression
        if temporal_expr.time_context:
            timestamp = f"{timestamp} in the {temporal_expr.time_context}"

        temporally_parsed_seed_memory = re.sub(r'\((\d{2}):(\d{2})\s*\[(\d{2}/\d{2}/\d{2})\]\)', 
            lambda m: f"({self.temporal_parser.get_temporal_expression(datetime.strptime(f'{m.group(1)}:{m.group(2)} {m.group(3)}', '%H:%M %d/%m/%y')).base_expression})", 
            seed_memory)

        prompt = self.prompt_formats['generate_dmn_thought'].format(
            memory_text=memory_context,
            seed_memory=temporally_parsed_seed_memory,
            timestamp=timestamp,  # Using natural language timestamp
            user_name=user_name,
            amygdala_response=self.amygdala_response
        )
        
        # personality temperature scaling based on memory density relative to top_k
        num_results = len(related_memories)

        if int(self.top_k) > 0:
            # Calculate inverse density: fewer memories = higher multiplier
            # Cap density at 1.0 to prevent going below 0.0
            density = min(1.0, num_results / max(1, int(self.top_k)))
            intensity_multiplier = 1.0 - density
        else:
            # Default to max intensity if top_k is 0 (edge case)
            # Corresponds to density = 0 in the formula
            density = 0.0
            intensity_multiplier = 1.0 
            
        # Calculate final intensity using the exact original clamping/scaling
        new_intensity = min(100, max(0, int(100 * intensity_multiplier)))
        # intensity already computed above as new_intensity and density already computed
        self.amygdala_response=new_intensity
        intensity_norm=new_intensity/100.0
        self.temperature=0.3+0.7*intensity_norm  # range [0.3, 1.0] — matches provider ceiling
        self.runtime.amygdala_response=new_intensity
        # Convert intensity to temperature before passing to API client
        self.runtime.update_api_temperature(self.temperature)

        top_p_value=0.98 if density<.33 else 0.95 if density<.66 else 0.92
        self.runtime.update_api_top_p(top_p_value)

        self.logger.info(f"Updated bot amygdala arousal to {new_intensity} based on memory density")
        self.logger.info(f"Updated bot top_p to {top_p_value:.2f} (banded density mapping)")

        system_prompt = self.system_prompts['dmn_thought_generation'].replace(
            '{amygdala_response}',
            str(self.amygdala_response)
        )
        # truncate the middle of each memory in the memory_context using truncate_middle
        memory_context = "\n\n".join([truncate_middle(memory, self.max_memory_length) for memory in memory_context.split("\n\n")])

        try:
            if self.use_chronpression and _chronomic_filter is not None:
                chron_compression = 0.5 + intensity_norm * (self.chron_compression_max - 0.5)
                raw_texts = seed_memory + "\n\n" + "\n\n".join(m for m, _ in related_memories)
                new_thought = _chronomic_filter(raw_texts, compression=chron_compression).strip()
                self.logger.info(f"dmn.chronpression amygdala={new_intensity} compression={chron_compression:.3f} [{len(raw_texts)}→{len(new_thought)}ch] >> {new_thought}")
            else:
                # Use call_api with override parameters without changing global state
                api_kwargs = {
                    'prompt': prompt,
                    'system_prompt': system_prompt,
                    'temperature': self.temperature
                }

                # Only add overrides if they're actually set
                if self.dmn_api_type:
                    api_kwargs['api_type_override'] = self.dmn_api_type
                if self.dmn_model:
                    api_kwargs['model_override'] = self.dmn_model

                new_thought = await self.runtime.call_api(**api_kwargs)
                new_thought, thinking_traces = separate_thinking_traces(new_thought)
                await store_thinking_traces(
                    self.memory_index,
                    user_id,
                    user_name,
                    thinking_traces,
                )
                new_thought = clean_response(new_thought)
            
            # Gather unique users from related memories
            memory_users = set()
            for memory, _ in related_memories:
                memory_id = self.memory_index.memories.index(memory)
                memory_user_id = next((uid for uid, mems in self.memory_index.user_memories.items() if memory_id in mems), None)
                if memory_user_id:
                    try:
                        resolved = await self.runtime.resolve_user(memory_user_id)
                        if resolved != user_name:
                            memory_users.add(resolved)
                    except Exception:
                        continue
            
            # Save generated thought as new memory
            timestamp = datetime.now().strftime('(%H:%M [%d/%m/%y])')
            users_str = PROMPTS.users_suffix.format(users=', '.join(memory_users)) if memory_users else ""
            # Clean username - remove all possible leading Discord role markers
            clean_name = re.sub(r'^[.!~*$]', '', user_name).strip()
            label = PROMPTS.chronpression_label if self.use_chronpression else PROMPTS.llm_label
            thought_memory = PROMPTS.thought_memory.format(
                label=label, name=clean_name, users=users_str,
                timestamp=timestamp, thought=new_thought,
            )
            # Store memory without metadata
            await self.memory_index.add_memory_async(user_id, thought_memory)

            # Probabilistic seed decay: each term entry removed with p=decay_rate.
            # Expected visits to full disconnection ≈ ln(terms)/decay_rate (geometric survival).
            # decay_rate=1.0 reproduces old single-visit behaviour; default ~0.25 gives gradual wear.
            # Validate the carried seed_mid is still pointing at the right text (fix 6).
            if (seed_mid is not None and
                    seed_mid < len(self.memory_index.memories) and
                    self.memory_index.memories[seed_mid] == seed_memory):
                decayed = 0
                with self.memory_index._mut:
                    for term in list(self.memory_index.inverted_index.keys()):
                        pl = self.memory_index.inverted_index[term]
                        if seed_mid not in pl:
                            continue
                        if random.random() >= self.decay_rate:
                            continue
                        new_pl = list(pl)
                        new_pl.remove(seed_mid)
                        decayed += 1
                        if new_pl:
                            self.memory_index.inverted_index[term] = new_pl
                        else:
                            del self.memory_index.inverted_index[term]
                self.memory_index._saver.request()
                self.logger.info(f"dmn.seed_decay mid={seed_mid} p={self.decay_rate} terms_decayed={decayed}")
            else:
                self.logger.info(f"dmn.seed_decay skipped: mid={seed_mid} stale or missing")

            # Similarity-graded neighbour thinning: close neighbours (high s) are thinned
            # harder because their gist was just captured in the distillation child.
            # Distant neighbours contributed novelty — their shared terms are the only bridge
            # keeping them reachable, so we thin much more gently.
            # p(remove one term entry) = thin_p_min + (thin_p_max - thin_p_min) * s
            # Removal is tf-1 (single list.remove), never wholesale.
            if 'memory_update_state' in locals():
                state = memory_update_state
                pmin = self.thin_p_min; pmax = self.thin_p_max
                thinned = {}
                with self.memory_index._mut:
                    for memory, memory_id in state['top_memories'][1:]:
                        # Stale-mid guard (fix 6): skip if index shifted under us
                        if (memory_id >= len(self.memory_index.memories) or
                                self.memory_index.memories[memory_id] != memory):
                            continue
                        s = state['score_by_mid'].get(memory_id, 0.0)
                        p = pmin + (pmax - pmin) * max(0.0, min(1.0, s))
                        shared = state['memory_terms_map'][memory_id] & state['overlapping_terms']
                        removed = []
                        for term in shared:
                            if random.random() >= p:
                                continue
                            pl = self.memory_index.inverted_index.get(term)
                            if not pl or memory_id not in pl:
                                continue
                            new_pl = list(pl)
                            new_pl.remove(memory_id)
                            removed.append(term)
                            if new_pl:
                                self.memory_index.inverted_index[term] = new_pl
                            else:
                                del self.memory_index.inverted_index[term]
                        if removed:
                            thinned[memory_id] = {'score': round(s, 3), 'p': round(p, 3), 'terms': removed}
                if thinned:
                    self.memory_index._saver.request()
                    self.logger.info(f"dmn.thin neighbours={len(thinned)}")
                    self.logger.log({'event': 'dmn_neighbour_thinning', 'timestamp': datetime.now().isoformat(), 'thinned': {str(k): v for k, v in thinned.items()}})

            # Add cleanup here after new memory addition and weight updates
            self._cleanup_disconnected_memories()
            # Log successful thought generation
            self.logger.log({
                'event': 'dmn_thought_generated',
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'user_name': user_name,
                'seed_memory': seed_memory,
                'system_prompt': system_prompt,
                'prompt': prompt,
                'generated_thought': new_thought,
                'amygdala_response': self.amygdala_response,
                'temperature': self.temperature
            })
            
        except Exception as e:
            error_msg = f"Failed to generate DMN thought: {str(e)}"
            self.logger.error(error_msg)
            # Log error in thought generation
            self.logger.log({
                'event': 'dmn_thought_error',
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'user_name': user_name,
                'error': str(e)
            })

    def _cleanup_disconnected_memories(self):
        with self.memory_index._mut:
            connected=set()
            for v in self.memory_index.inverted_index.values():connected.update(v)
            disc=[i for i,m in enumerate(self.memory_index.memories) if m is not None and i not in connected]
            if not disc:return
            texts=[self.memory_index.memories[i] for i in disc]
            owners={}
            for uid,mems in self.memory_index.user_memories.items():
                for mid in mems:
                    if mid in connected:continue
                    owners.setdefault(mid,uid)
            for i in disc:self.memory_index.memories[i]=None
            self.memory_index._compact()
        self.memory_index._saver.request()
        self.logger.info(f"dmn.cleanup removed={len(disc)}")
        self.logger.log({'event':'dmn_memory_cleanup','timestamp':datetime.now().isoformat(),'removed':len(disc),'owners':{str(k):v for k,v in owners.items()},'disconnected_memories':texts})


    def set_mode(self, mode):
        """Update DMN parameters based on mode."""
        m = {k.lower(): k for k in self.modes}
        key = str(mode).strip().lower()
        if key not in m:
            raise ValueError(f"unknown mode: {mode}")
        p = self.modes[m[key]]
        self.combination_threshold = float(p["combination_threshold"])
        self.similarity_threshold = float(p["similarity_threshold"])
        self.decay_rate = float(p["decay_rate"])
        self.top_k = int(p["top_k"])
        self.fuzzy_overlap_threshold = int(p["fuzzy_overlap_threshold"])
        self.fuzzy_search_threshold = int(p["fuzzy_search_threshold"])
        self.thin_p_min = float(p.get("thin_p_min", self.thin_p_min))
        self.thin_p_max = float(p.get("thin_p_max", self.thin_p_max))
