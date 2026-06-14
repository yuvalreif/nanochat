use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use serde::Deserialize;
use serde_json::json;
use std::collections::{HashMap, HashSet};
use tiktoken_rs::CoreBPE;
use tokenizers::Tokenizer;

const SURFACE_GROUP_NAMES: [&str; 2] = ["space_prefix", "base_capitalization"];
const STRIP_NONLEXICAL_GROUPS: [&str; 6] = [
    "determiners",
    "prepositions",
    "article_capitalization",
    "prep_capitalization",
    "prefix_punctuation",
    "suffix_punctuation",
];
const STRIP_INVALID_DETACHED_GROUPS: [&str; 6] = [
    "determiners",
    "prepositions",
    "article_capitalization",
    "prep_capitalization",
    "article_space_prefix",
    "prep_space_prefix",
];

#[allow(dead_code)]
#[derive(Debug, Deserialize, Clone)]
struct Config {
    version: usize,
    num_modifier_groups: usize,
    default_modifier: Vec<u16>,
    #[serde(default)]
    group_names: Vec<String>,
    #[serde(default)]
    group_value_names: HashMap<String, Vec<String>>,
    entries: Vec<Entry>,
    #[serde(default)]
    reverse_entries: Vec<ReverseEntry>,
    #[serde(default)]
    token_meta: Vec<TokenMeta>,
    #[serde(default)]
    runtime: RuntimeConfig,
    #[serde(default)]
    base_bpe: Option<BaseBpeConfig>,
    tokenizer_json: Option<String>,
}

#[derive(Debug, Deserialize, Clone, Default)]
struct BaseBpeConfig {
    #[serde(default)]
    pattern: String,
    #[serde(default)]
    mergeable_ranks: Vec<BaseBpeRank>,
    #[serde(default)]
    special_tokens: HashMap<String, u32>,
}

#[derive(Debug, Deserialize, Clone, Default)]
struct BaseBpeRank {
    #[serde(default)]
    token: String,
    #[serde(default)]
    rank: u32,
}

#[derive(Debug, Deserialize, Clone)]
struct Entry {
    token_ids: Vec<u32>,
    base_ids: Vec<u32>,
    modifier_rows: Vec<Vec<u16>>,
}

#[derive(Debug, Deserialize, Clone)]
struct ReverseEntry {
    token_ids: Vec<u32>,
    base_ids: Vec<u32>,
    modifier_rows: Vec<Vec<u16>>,
    #[serde(default)]
    surface: Option<String>,
}

#[derive(Debug, Deserialize, Clone, Default)]
struct LiteralTransform {
    #[serde(default)]
    group_name: String,
    #[serde(default)]
    rel_idx: usize,
}

#[derive(Debug, Deserialize, Clone, Default)]
struct TokenMeta {
    #[serde(default)]
    token_text: String,
    #[serde(default)]
    canonical_surface: String,
    #[serde(default)]
    has_space_prefix: bool,
    #[serde(default)]
    has_word_char: bool,
    #[serde(default)]
    is_whitespace_only: bool,
    #[serde(default)]
    is_single_ascii_space: bool,
    #[serde(default)]
    is_byte_fallback: bool,
    #[serde(default)]
    is_base_cap_representable: bool,
    #[serde(default)]
    determiner: Option<LiteralTransform>,
    #[serde(default)]
    preposition: Option<LiteralTransform>,
    #[serde(default)]
    prefix_punctuation: Option<LiteralTransform>,
    #[serde(default)]
    suffix_punctuation: Option<LiteralTransform>,
}

#[derive(Debug, Deserialize, Clone)]
struct AttachmentLimits {
    #[serde(default = "default_attach_limit")]
    max_prefix_punctuation: usize,
    #[serde(default = "default_attach_limit")]
    max_suffix_punctuation: usize,
}

impl Default for AttachmentLimits {
    fn default() -> Self {
        Self {
            max_prefix_punctuation: default_attach_limit(),
            max_suffix_punctuation: default_attach_limit(),
        }
    }
}

#[derive(Debug, Deserialize, Clone, Default)]
struct RuntimeConfig {
    #[serde(default)]
    group_indices: HashMap<String, usize>,
    #[serde(default)]
    literal_maps: HashMap<String, HashMap<String, LiteralTransform>>,
    #[serde(default)]
    multi_token_first_group_indices: Vec<usize>,
    #[serde(default)]
    attachment_limits: AttachmentLimits,
}

#[derive(Clone)]
struct EntryValue {
    consumed_len: usize,
    base_ids: Vec<u32>,
    modifier_rows: Vec<Vec<u16>>,
}

#[derive(Clone)]
struct ReverseEntryValue {
    consumed_len: usize,
    base_ids: Vec<u32>,
    modifier_rows: Vec<Vec<u16>>,
    surface: Option<String>,
}

#[derive(Default)]
struct TrieNode {
    children: HashMap<u32, usize>,
    value: Option<EntryValue>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ReverseKey {
    token_id: u32,
    modifier_row: Vec<u16>,
}

#[derive(Default)]
struct ReverseTrieNode {
    children: HashMap<ReverseKey, usize>,
    value: Option<ReverseEntryValue>,
}

#[derive(Debug, Clone)]
struct PendingGroup {
    group_idx: usize,
    rel_idx: u16,
}

struct BaseBpeRuntime {
    core: CoreBPE,
    id_to_token_bytes: Vec<Option<Vec<u8>>>,
}

#[pyclass]
struct CompositionalProcessor {
    base_bpe: Option<BaseBpeRuntime>,
    tokenizer: Option<Tokenizer>,
    trie_nodes: Vec<TrieNode>,
    max_sequence_len: usize,
    reverse_trie_nodes: Vec<ReverseTrieNode>,
    max_reverse_sequence_len: usize,
    default_modifier: Vec<u16>,
    num_modifier_groups: usize,
    group_names: Vec<String>,
    group_value_names: HashMap<String, Vec<String>>,
    runtime: RuntimeConfig,
    token_meta: Vec<TokenMeta>,
}

fn default_attach_limit() -> usize {
    1
}

fn bytes_to_latin1(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| char::from(*byte)).collect()
}

fn latin1_to_bytes(text: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(text.len());
    for ch in text.chars() {
        let code = ch as u32;
        if code > 0xFF {
            return None;
        }
        out.push(code as u8);
    }
    Some(out)
}

fn decode_utf8_lossy(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).to_string()
}

fn first_alpha_is_upper(text: &str) -> Option<bool> {
    text.chars().find(|ch| ch.is_alphabetic()).map(|ch| ch.is_uppercase())
}

fn capitalize_first_alpha(text: &str) -> String {
    let mut out = String::new();
    let mut changed = false;
    for ch in text.chars() {
        if !changed && ch.is_alphabetic() {
            for upper in ch.to_uppercase() {
                out.push(upper);
            }
            changed = true;
        } else {
            out.push(ch);
        }
    }
    out
}

fn lowercase_first_alpha(text: &str) -> String {
    let mut out = String::new();
    let mut changed = false;
    for ch in text.chars() {
        if !changed && ch.is_alphabetic() {
            for lower in ch.to_lowercase() {
                out.push(lower);
            }
            changed = true;
        } else {
            out.push(ch);
        }
    }
    out
}

fn is_capitalized_surface(text: &str) -> bool {
    let stripped = text.trim_start_matches(' ');
    stripped
        .chars()
        .find(|ch| ch.is_alphabetic())
        .map(|ch| ch.is_uppercase())
        .unwrap_or(false)
}

fn is_base_cap_representable_surface(text: &str) -> bool {
    let alpha_chars: Vec<char> = text.chars().filter(|ch| ch.is_alphabetic()).collect();
    if alpha_chars.is_empty() {
        return true;
    }
    if alpha_chars.iter().all(|ch| ch.is_lowercase()) {
        return true;
    }
    alpha_chars[0].is_uppercase() && alpha_chars.iter().skip(1).all(|ch| ch.is_lowercase())
}

fn split_camel_case_segments(surface: &str) -> Option<Vec<String>> {
    if surface.is_empty() {
        return None;
    }
    let chars: Vec<char> = surface.chars().collect();
    let mut boundaries = vec![0usize];
    let mut has_internal_upper = false;
    for idx in 1..chars.len() {
        let current = chars[idx];
        if !current.is_uppercase() {
            continue;
        }
        has_internal_upper = true;
        let prev = chars[idx - 1];
        let next_is_lower = (idx + 1) < chars.len() && chars[idx + 1].is_lowercase();
        if prev.is_lowercase() || (prev.is_uppercase() && next_is_lower) {
            boundaries.push(idx);
        }
    }
    if !has_internal_upper {
        return None;
    }
    boundaries.push(chars.len());
    if boundaries.len() <= 2 {
        return None;
    }
    let mut segments = Vec::new();
    for pair in boundaries.windows(2) {
        let left = pair[0];
        let right = pair[1];
        if left < right {
            segments.push(chars[left..right].iter().collect());
        }
    }
    if segments.len() > 1 {
        Some(segments)
    } else {
        None
    }
}

fn expand_caps_segments(segments: Vec<String>) -> Vec<String> {
    let mut expanded = Vec::new();
    for segment in segments {
        if segment.len() > 1 && segment.chars().all(|ch| ch.is_uppercase()) {
            expanded.extend(segment.chars().map(|ch| ch.to_string()));
        } else {
            expanded.push(segment);
        }
    }
    expanded
}

impl CompositionalProcessor {
    fn add_entry(&mut self, entry: &Entry) {
        let mut node_idx = 0usize;
        for token_id in &entry.token_ids {
            let child_idx = if let Some(idx) = self.trie_nodes[node_idx].children.get(token_id) {
                *idx
            } else {
                let idx = self.trie_nodes.len();
                self.trie_nodes.push(TrieNode::default());
                self.trie_nodes[node_idx].children.insert(*token_id, idx);
                idx
            };
            node_idx = child_idx;
        }
        self.trie_nodes[node_idx].value = Some(EntryValue {
            consumed_len: entry.token_ids.len(),
            base_ids: entry.base_ids.clone(),
            modifier_rows: entry.modifier_rows.clone(),
        });
        self.max_sequence_len = self.max_sequence_len.max(entry.token_ids.len());
    }

    fn add_reverse_entry(&mut self, entry: &ReverseEntry) {
        let mut node_idx = 0usize;
        for (base_id, modifier_row) in entry.base_ids.iter().zip(entry.modifier_rows.iter()) {
            let key = ReverseKey {
                token_id: *base_id,
                modifier_row: modifier_row.clone(),
            };
            let child_idx = if let Some(idx) = self.reverse_trie_nodes[node_idx].children.get(&key) {
                *idx
            } else {
                let idx = self.reverse_trie_nodes.len();
                self.reverse_trie_nodes.push(ReverseTrieNode::default());
                self.reverse_trie_nodes[node_idx].children.insert(key, idx);
                idx
            };
            node_idx = child_idx;
        }
        self.reverse_trie_nodes[node_idx].value = Some(ReverseEntryValue {
            consumed_len: entry.base_ids.len(),
            base_ids: entry.base_ids.clone(),
            modifier_rows: entry.modifier_rows.clone(),
            surface: entry.surface.clone(),
        });
        self.max_reverse_sequence_len = self.max_reverse_sequence_len.max(entry.base_ids.len());
    }

    fn group_idx(&self, group_name: &str) -> Option<usize> {
        self.runtime.group_indices.get(group_name).copied()
    }

    fn empty_modifier(&self) -> Vec<u16> {
        self.default_modifier.clone()
    }

    fn token_meta_ref(&self, token_id: u32) -> &TokenMeta {
        self.token_meta.get(token_id as usize).unwrap_or(&self.token_meta[0])
    }

    fn raw_span_has_byte_fallback(&self, raw_ids: &[u32], start_idx: usize, consumed_len: usize) -> bool {
        let end_idx = usize::min(start_idx + consumed_len, raw_ids.len());
        raw_ids[start_idx..end_idx]
            .iter()
            .any(|token_id| self.token_meta_ref(*token_id).is_byte_fallback)
    }

    fn base_bpe_token_bytes(&self, token_id: u32) -> Option<&[u8]> {
        self.base_bpe
            .as_ref()
            .and_then(|runtime| runtime.id_to_token_bytes.get(token_id as usize))
            .and_then(|opt| opt.as_deref())
    }

    fn decode_token_bytes(&self, token_ids: &[u32]) -> Option<String> {
        let mut bytes = Vec::new();
        for token_id in token_ids {
            bytes.extend_from_slice(self.base_bpe_token_bytes(*token_id)?);
        }
        std::str::from_utf8(&bytes).ok().map(|text| text.to_string())
    }

    fn byte_component_end(&self, token_ids: &[u32], start_idx: usize) -> usize {
        if start_idx >= token_ids.len() {
            return start_idx;
        }
        let Some(first_bytes) = self.base_bpe_token_bytes(token_ids[start_idx]) else {
            return start_idx + 1;
        };
        if std::str::from_utf8(first_bytes).is_ok() {
            return start_idx + 1;
        }
        let mut pending = Vec::new();
        for end_idx in start_idx..token_ids.len() {
            let Some(token_bytes) = self.base_bpe_token_bytes(token_ids[end_idx]) else {
                break;
            };
            pending.extend_from_slice(token_bytes);
            if std::str::from_utf8(&pending).is_ok() {
                return end_idx + 1;
            }
        }
        start_idx + 1
    }

    fn byte_component_has_word_char(&self, token_ids: &[u32], start_idx: usize) -> bool {
        let end_idx = self.byte_component_end(token_ids, start_idx);
        self.decode_token_bytes(&token_ids[start_idx..end_idx])
            .map(|text| text.chars().any(|ch| ch.is_alphanumeric()))
            .unwrap_or(false)
    }

    fn raw_position_has_word_char(&self, raw_ids: &[u32], idx: usize) -> bool {
        if idx >= raw_ids.len() {
            return false;
        }
        let meta = self.token_meta_ref(raw_ids[idx]);
        if meta.has_word_char {
            return true;
        }
        meta.is_byte_fallback && self.byte_component_has_word_char(raw_ids, idx)
    }

    fn modifier_utf8_delta(&self, modifiers: &[u16]) -> Option<usize> {
        for (group_idx, value) in modifiers.iter().enumerate() {
            if *value == self.default_modifier[group_idx] {
                continue;
            }
            let group_name = self.group_names.get(group_idx)?;
            match group_name.as_str() {
                "space_prefix"
                | "base_capitalization"
                | "determiners"
                | "article_det"
                | "articles"
                | "article_capitalization"
                | "article_space_prefix"
                | "prepositions"
                | "prep_capitalization"
                | "prep_space_prefix"
                | "prefix_punctuation"
                | "suffix_punctuation" => {}
                _ => return None,
            }
        }

        let mut delta = 0usize;
        if self.space_setting(modifiers, "space_prefix") {
            delta += 1;
        }
        if let Some(value) = self.literal_from_group(modifiers, "prepositions", &["prep_"]) {
            delta += value.as_bytes().len() + 1;
        }
        let determiner = self
            .literal_from_group(modifiers, "determiners", &["det_", "article_"])
            .or_else(|| self.literal_from_group(modifiers, "article_det", &["det_", "article_"]))
            .or_else(|| self.literal_from_group(modifiers, "articles", &["article_", "det_"]));
        if let Some(value) = determiner {
            delta += value.as_bytes().len() + 1;
        }
        if let Some(value) = self.literal_from_group(modifiers, "prefix_punctuation", &["punct_prefix_"]) {
            delta += value.as_bytes().len();
        }
        if let Some(value) = self.literal_from_group(modifiers, "suffix_punctuation", &["punct_suffix_"]) {
            delta += value.as_bytes().len();
        }
        Some(delta)
    }

    fn base_bpe_vocab_token(&self, token_id: u32) -> Option<String> {
        self.base_bpe_token_bytes(token_id).map(bytes_to_latin1)
    }

    fn base_bpe_decoded_token(&self, token_id: u32) -> Option<String> {
        let runtime = self.base_bpe.as_ref()?;
        if let Ok(text) = runtime.core.decode(&[token_id]) {
            return Some(text);
        }
        runtime
            .core
            .decode_bytes(&[token_id])
            .ok()
            .map(|bytes| decode_utf8_lossy(&bytes))
    }

    fn decode_ids(&self, ids: &[u32]) -> String {
        if let Some(runtime) = self.base_bpe.as_ref() {
            if let Ok(text) = runtime.core.decode(ids) {
                return text;
            }
            if let Ok(bytes) = runtime.core.decode_bytes(ids) {
                return decode_utf8_lossy(&bytes);
            }
            return String::new();
        }
        self.tokenizer
            .as_ref()
            .and_then(|tok| tok.decode(ids, false).ok())
            .unwrap_or_default()
    }

    fn decode_single(&self, token_id: u32) -> String {
        self.decode_ids(&[token_id])
    }

    fn encode_segment(&self, text: &str) -> Option<Vec<u32>> {
        if let Some(runtime) = self.base_bpe.as_ref() {
            return Some(runtime.core.encode_ordinary(text));
        }
        self.tokenizer
            .as_ref()
            .and_then(|tok| tok.encode(text, false).ok())
            .map(|encoding| encoding.get_ids().iter().map(|v| *v as u32).collect())
    }

    fn longest_reverse_match(&self, token_ids: &[u32], modifier_rows: &[Vec<u16>], start_idx: usize) -> Option<ReverseEntryValue> {
        let mut node_idx = 0usize;
        let mut best: Option<ReverseEntryValue> = None;
        let stop = usize::min(start_idx + self.max_reverse_sequence_len, token_ids.len());
        for pos in start_idx..stop {
            let key = ReverseKey {
                token_id: token_ids[pos],
                modifier_row: modifier_rows[pos].clone(),
            };
            let child_idx = match self.reverse_trie_nodes[node_idx].children.get(&key) {
                Some(idx) => *idx,
                None => break,
            };
            node_idx = child_idx;
            if let Some(value) = self.reverse_trie_nodes[node_idx].value.clone() {
                best = Some(value);
            }
        }
        best
    }

    fn value_name(&self, group_name: &str, value: u16) -> Option<String> {
        self.group_value_names
            .get(group_name)
            .and_then(|names| names.get(value as usize))
            .cloned()
    }

    fn literal_from_group(&self, modifiers: &[u16], group_name: &str, prefixes: &[&str]) -> Option<String> {
        let group_idx = self.group_idx(group_name)?;
        let value = *modifiers.get(group_idx)?;
        if value == self.default_modifier[group_idx] {
            return None;
        }
        let value_name = self.value_name(group_name, value)?;
        for prefix in prefixes {
            if value_name.starts_with(prefix) {
                return Some(value_name[prefix.len()..].to_string());
            }
        }
        None
    }

    fn space_setting(&self, modifiers: &[u16], group_name: &str) -> bool {
        let Some(group_idx) = self.group_idx(group_name) else {
            return false;
        };
        let value = modifiers[group_idx];
        if value == self.default_modifier[group_idx] {
            return false;
        }
        let value_name = self.value_name(group_name, value).unwrap_or_default().to_lowercase();
        if value_name.starts_with("with_") || value_name.starts_with("add_") {
            return true;
        }
        if value_name.starts_with("remove_")
            || value_name.starts_with("lower_")
            || value_name.starts_with("no_")
            || value_name.starts_with("na_")
            || value_name.starts_with("none")
        {
            return false;
        }
        value == 1
    }

    fn apply_base_capitalization(&self, surface: &str, modifiers: &[u16]) -> String {
        let Some(group_idx) = self.group_idx("base_capitalization") else {
            return surface.to_string();
        };
        let value = modifiers[group_idx];
        if value == self.default_modifier[group_idx] {
            return surface.to_string();
        }
        let value_name = self
            .value_name("base_capitalization", value)
            .unwrap_or_default()
            .to_lowercase();
        if value_name.starts_with("add_") || value_name.starts_with("with_") || value == 1 {
            return capitalize_first_alpha(surface);
        }
        if value_name.starts_with("remove_") || value_name.starts_with("lower_") {
            return lowercase_first_alpha(surface);
        }
        surface.to_string()
    }

    fn synthesize_surface(&self, lexical_surface: &str, modifiers: &[u16]) -> String {
        let mut surface = if lexical_surface.is_empty() {
            String::new()
        } else if lexical_surface.trim().is_empty() {
            lexical_surface.to_string()
        } else {
            lexical_surface.trim_start_matches(' ').to_string()
        };
        surface = self.apply_base_capitalization(&surface, modifiers);

        let prefix_punct = self.literal_from_group(modifiers, "prefix_punctuation", &["punct_prefix_"]);
        let mut preposition = self.literal_from_group(modifiers, "prepositions", &["prep_"]);
        let mut determiner = self.literal_from_group(modifiers, "determiners", &["det_", "article_"])
            .or_else(|| self.literal_from_group(modifiers, "article_det", &["det_", "article_"]))
            .or_else(|| self.literal_from_group(modifiers, "articles", &["article_", "det_"]));
        let suffix_punct = self.literal_from_group(modifiers, "suffix_punctuation", &["punct_suffix_"]);

        if let Some(text) = preposition.as_ref() {
            if self.space_setting(modifiers, "prep_capitalization") {
                preposition = Some(capitalize_first_alpha(text));
            }
        }
        if let Some(text) = determiner.as_ref() {
            if self.space_setting(modifiers, "article_capitalization") {
                determiner = Some(capitalize_first_alpha(text));
            }
        }

        let mut pieces = Vec::new();
        if let Some(value) = preposition {
            if !value.is_empty() {
                pieces.push(value);
            }
        }
        if let Some(value) = determiner {
            if !value.is_empty() {
                pieces.push(value);
            }
        }
        if !surface.is_empty() {
            pieces.push(surface);
        }

        let mut expr = pieces.join(" ");
        if let Some(value) = prefix_punct {
            expr = format!("{value}{expr}");
        }
        if let Some(value) = suffix_punct {
            expr = format!("{expr}{value}");
        }
        if self.space_setting(modifiers, "space_prefix") && !expr.is_empty() && !expr.starts_with(char::is_whitespace) {
            expr.insert(0, ' ');
        }
        expr
    }

    fn combine_modifier_rows(&self, modifier_rows: &[Vec<u16>]) -> Vec<u16> {
        let mut combined = self.default_modifier.clone();
        for row in modifier_rows {
            for (group_idx, value) in row.iter().enumerate() {
                if *value != self.default_modifier[group_idx] {
                    combined[group_idx] = *value;
                }
            }
        }
        combined
    }

    fn combine_pending(&self, base_modifier: &[u16], pending_groups: &[PendingGroup]) -> Vec<u16> {
        if pending_groups.is_empty() {
            return base_modifier.to_vec();
        }
        let mut combined = base_modifier.to_vec();
        let mut applied = HashSet::new();
        for group in pending_groups.iter().rev() {
            if applied.insert(group.group_idx) {
                combined[group.group_idx] = group.rel_idx;
            }
        }
        combined
    }

    fn spread_multi_token_modifiers(&self, combined_modifier: &[u16], base_len: usize) -> Vec<Vec<u16>> {
        if base_len <= 1 {
            return vec![combined_modifier.to_vec()];
        }
        let first_groups: HashSet<usize> = self.runtime.multi_token_first_group_indices.iter().copied().collect();
        let mut first_modifier = self.empty_modifier();
        let mut last_modifier = self.empty_modifier();
        for (idx, value) in combined_modifier.iter().enumerate() {
            if *value == self.default_modifier[idx] {
                continue;
            }
            if first_groups.contains(&idx) {
                first_modifier[idx] = *value;
            } else {
                last_modifier[idx] = *value;
            }
        }
        if base_len == 2 {
            return vec![first_modifier, last_modifier];
        }
        let mut out = vec![first_modifier];
        for _ in 0..(base_len - 2) {
            out.push(self.empty_modifier());
        }
        out.push(last_modifier);
        out
    }

    fn modifier_has_active_group(&self, modifier: &[u16], group_name: &str) -> bool {
        let Some(group_idx) = self.group_idx(group_name) else {
            return false;
        };
        modifier[group_idx] != self.default_modifier[group_idx]
    }

    fn modifier_has_only_surface_groups(&self, modifier: &[u16]) -> bool {
        let mut allowed = HashSet::new();
        for group_name in SURFACE_GROUP_NAMES {
            if let Some(group_idx) = self.group_idx(group_name) {
                allowed.insert(group_idx);
            }
        }
        for (idx, value) in modifier.iter().enumerate() {
            if *value == self.default_modifier[idx] {
                continue;
            }
            if !allowed.contains(&idx) {
                return false;
            }
        }
        true
    }

    fn is_intra_word_cap_alias_match(&self, match_length: usize, modifier: &[u16]) -> bool {
        if match_length != 1 {
            return false;
        }
        let Some(group_idx) = self.group_idx("base_capitalization") else {
            return false;
        };
        modifier[group_idx] == 1 && self.modifier_has_only_surface_groups(modifier)
    }

    fn find_longest_boundary_safe_match(
        &self,
        raw_ids: &[u32],
        start_idx: usize,
        space_prefix_prefix_sum: &[usize],
    ) -> Option<EntryValue> {
        if start_idx >= raw_ids.len() {
            return None;
        }
        let start_inside_word = start_idx > 0
            && self.token_meta_ref(raw_ids[start_idx]).has_word_char
            && !self.token_meta_ref(raw_ids[start_idx]).has_space_prefix
            && self.token_meta_ref(raw_ids[start_idx - 1]).has_word_char;
        let mut node_idx = 0usize;
        let max_end = usize::min(start_idx + self.max_sequence_len, raw_ids.len());
        let mut best: Option<EntryValue> = None;
        let mut best_length = 0usize;
        for end_idx in start_idx..max_end {
            let token_id = raw_ids[end_idx];
            let child = match self.trie_nodes[node_idx].children.get(&token_id) {
                Some(idx) => *idx,
                None => break,
            };
            node_idx = child;
            let Some(entry) = self.trie_nodes[node_idx].value.clone() else {
                continue;
            };
            let span_end = end_idx + 1;
            let match_length = span_end - start_idx;
            let combined_modifier = self.combine_modifier_rows(&entry.modifier_rows);
            let allow_intra_word_cap_alias = self.is_intra_word_cap_alias_match(match_length, &combined_modifier);
            if start_inside_word && !allow_intra_word_cap_alias {
                continue;
            }
            if span_end < raw_ids.len()
                && self.token_meta_ref(raw_ids[end_idx]).has_word_char
                && !self.token_meta_ref(raw_ids[span_end]).has_space_prefix
                && self.token_meta_ref(raw_ids[span_end]).has_word_char
                && !allow_intra_word_cap_alias
            {
                continue;
            }
            if match_length > 1 {
                let all_word = (start_idx..span_end).all(|j| self.token_meta_ref(raw_ids[j]).has_word_char);
                if all_word && (space_prefix_prefix_sum[span_end] - space_prefix_prefix_sum[start_idx + 1]) > 0 {
                    continue;
                }
            }
            best = Some(entry);
            best_length = match_length;
        }
        if best_length > 0 { best } else { None }
    }

    fn should_prefer_cap_fallback_over_match(&self, raw_ids: &[u32], start_idx: usize, entry: &EntryValue) -> bool {
        let match_length = entry.consumed_len;
        let modifier = self.combine_modifier_rows(&entry.modifier_rows);
        if !self.is_intra_word_cap_alias_match(match_length, &modifier) {
            return false;
        }
        if start_idx >= raw_ids.len() || !self.token_meta_ref(raw_ids[start_idx]).has_word_char {
            return false;
        }
        let prev_continues_word = start_idx > 0
            && self.token_meta_ref(raw_ids[start_idx - 1]).has_word_char
            && !self.token_meta_ref(raw_ids[start_idx]).has_space_prefix;
        let next_idx = start_idx + match_length;
        let next_continues_word = next_idx < raw_ids.len()
            && self.token_meta_ref(raw_ids[next_idx]).has_word_char
            && !self.token_meta_ref(raw_ids[next_idx]).has_space_prefix;
        prev_continues_word || next_continues_word
    }

    fn try_lowercase_cap_fallback(
        &self,
        raw_ids: &[u32],
        start_idx: usize,
        pending_groups: &[PendingGroup],
        pending_leading_space: bool,
    ) -> Option<(usize, Vec<u32>, Vec<Vec<u16>>)> {
        let base_cap_idx = self.group_idx("base_capitalization")?;
        if start_idx >= raw_ids.len() || !self.token_meta_ref(raw_ids[start_idx]).has_word_char {
            return None;
        }
        let mut end_idx = start_idx + 1;
        while end_idx < raw_ids.len() {
            let meta = self.token_meta_ref(raw_ids[end_idx]);
            if meta.has_space_prefix || !meta.has_word_char || meta.is_whitespace_only {
                break;
            }
            end_idx += 1;
        }
        let mut surface = self.decode_ids(&raw_ids[start_idx..end_idx]);
        surface = surface.trim().to_string();
        if surface.is_empty() || !surface.chars().all(|ch| ch.is_alphabetic()) || !surface.chars().any(|ch| ch.is_uppercase()) {
            return None;
        }
        if !is_base_cap_representable_surface(&surface) {
            return None;
        }
        let split_segments = split_camel_case_segments(&surface);
        let is_title_surface = surface.chars().next().map(|ch| ch.is_uppercase()).unwrap_or(false)
            && surface.chars().skip(1).all(|ch| !ch.is_alphabetic() || ch.is_lowercase());
        if split_segments.is_none() && !is_title_surface {
            return None;
        }
        let segments = expand_caps_segments(split_segments.unwrap_or_else(|| vec![surface]));
        let mut output_ids = Vec::new();
        let mut output_mods = Vec::new();
        let mut first_output = true;
        let space_idx = self.group_idx("space_prefix");
        for segment in segments {
            let lower_ids = self.encode_segment(&segment.to_lowercase())?;
            if lower_ids.is_empty() {
                return None;
            }
            let mut base_modifier = self.empty_modifier();
            base_modifier[base_cap_idx] = 1;
            if first_output {
                if pending_leading_space {
                    if let Some(idx) = space_idx {
                        base_modifier[idx] = 1;
                    }
                }
                base_modifier = self.combine_pending(&base_modifier, pending_groups);
            }
            let per_token_mods = self.spread_multi_token_modifiers(&base_modifier, lower_ids.len());
            output_ids.extend(lower_ids);
            output_mods.extend(per_token_mods);
            first_output = false;
        }
        Some((end_idx - start_idx, output_ids, output_mods))
    }

    fn can_attach_detached_modifier(
        &self,
        raw_ids: &[u32],
        start_idx: usize,
        consumed_len: usize,
        pending_has_prefix_punct: bool,
    ) -> bool {
        if consumed_len == 0 || start_idx >= raw_ids.len() {
            return false;
        }
        let span_end = usize::min(start_idx + consumed_len, raw_ids.len());
        let left_ok = if start_idx == 0 {
            true
        } else {
            self.token_meta_ref(raw_ids[start_idx]).has_space_prefix
                || self.token_meta_ref(raw_ids[start_idx - 1]).is_whitespace_only
        };
        if !left_ok && !pending_has_prefix_punct {
            return false;
        }
        let mut j = span_end;
        let mut saw_whitespace_between = false;
        while j < raw_ids.len() && self.token_meta_ref(raw_ids[j]).is_whitespace_only {
            if !self.token_meta_ref(raw_ids[j]).is_single_ascii_space {
                return false;
            }
            saw_whitespace_between = true;
            j += 1;
        }
        if j >= raw_ids.len() || !self.raw_position_has_word_char(raw_ids, j) {
            return false;
        }
        let next_surface = &self.token_meta_ref(raw_ids[j]).canonical_surface;
        let current_surface = &self.token_meta_ref(raw_ids[start_idx]).canonical_surface;
        let has_prep = self.runtime.literal_maps.get("prepositions").map(|m| m.contains_key(next_surface)).unwrap_or(false);
        if has_prep {
            return false;
        }
        let current_is_prep = self.runtime.literal_maps.get("prepositions").map(|m| m.contains_key(current_surface)).unwrap_or(false);
        let next_is_det = self.runtime.literal_maps.get("determiners").map(|m| m.contains_key(next_surface)).unwrap_or(false);
        if current_is_prep && next_is_det {
            return true;
        }
        if next_is_det {
            return false;
        }
        saw_whitespace_between || self.token_meta_ref(raw_ids[j]).has_space_prefix
    }

    fn strip_groups(&self, modifier_values: &[u16], group_names: &[&str]) -> Vec<u16> {
        let mut cleaned = modifier_values.to_vec();
        for group_name in group_names {
            if let Some(group_idx) = self.group_idx(group_name) {
                cleaned[group_idx] = self.default_modifier[group_idx];
            }
        }
        cleaned
    }

    fn strip_invalid_detached_modifier_groups(
        &self,
        modifier_values: &[u16],
        raw_ids: &[u32],
        start_idx: usize,
        consumed_len: usize,
        pending_has_prefix_punct: bool,
    ) -> Vec<u16> {
        if self.can_attach_detached_modifier(raw_ids, start_idx, consumed_len, pending_has_prefix_punct) {
            return modifier_values.to_vec();
        }
        self.strip_groups(modifier_values, &STRIP_INVALID_DETACHED_GROUPS)
    }

    fn strip_nonlexical_surface_groups(&self, modifier_values: &[u16], base_ids: &[u32]) -> Vec<u16> {
        if base_ids.iter().any(|base_id| self.token_meta_ref(*base_id).has_word_char) {
            return modifier_values.to_vec();
        }
        self.strip_groups(modifier_values, &STRIP_NONLEXICAL_GROUPS)
    }

    fn raw_expr_has_leading_space(&self, raw_ids: &[u32], start_idx: usize) -> bool {
        if start_idx == 0 {
            return false;
        }
        let prev_token_id = raw_ids[start_idx - 1];
        if self.token_meta_ref(prev_token_id).is_whitespace_only {
            return false;
        }
        self.token_meta_ref(raw_ids[start_idx]).has_space_prefix
    }

    fn next_non_whitespace_idx(&self, raw_ids: &[u32], start_idx: usize) -> Option<usize> {
        let mut idx = start_idx;
        while idx < raw_ids.len() {
            if !self.token_meta_ref(raw_ids[idx]).is_whitespace_only {
                return Some(idx);
            }
            idx += 1;
        }
        None
    }

    fn token_can_host_expr_space(&self, raw_ids: &[u32], start_idx: usize) -> bool {
        let mut idx = start_idx;
        let mut saw_ascii_space = false;
        while idx < raw_ids.len() && self.token_meta_ref(raw_ids[idx]).is_whitespace_only {
            if !self.token_meta_ref(raw_ids[idx]).is_single_ascii_space {
                return false;
            }
            if saw_ascii_space {
                return false;
            }
            saw_ascii_space = true;
            idx += 1;
        }
        let Some(next_idx) = self.next_non_whitespace_idx(raw_ids, start_idx) else {
            return false;
        };
        let next_meta = self.token_meta_ref(raw_ids[next_idx]);
        if next_meta.has_space_prefix {
            return false;
        }
        if self.raw_position_has_word_char(raw_ids, next_idx) {
            return true;
        }
        if next_meta.determiner.is_some() || next_meta.preposition.is_some() {
            return true;
        }
        if next_meta.prefix_punctuation.is_some() {
            let lookahead_idx = self.next_non_whitespace_idx(raw_ids, next_idx + 1);
            return lookahead_idx.map(|idx| self.raw_position_has_word_char(raw_ids, idx)).unwrap_or(false);
        }
        false
    }

    fn apply_contextual_space_prefix(
        &self,
        modifier_values: &[u16],
        raw_ids: &[u32],
        start_idx: usize,
        use_pending_space: bool,
        pending_leading_space: bool,
    ) -> Vec<u16> {
        let Some(space_idx) = self.group_idx("space_prefix") else {
            return modifier_values.to_vec();
        };
        let mut normalized = modifier_values.to_vec();
        let expr_has_space = if use_pending_space {
            pending_leading_space
        } else {
            self.raw_expr_has_leading_space(raw_ids, start_idx)
        };
        normalized[space_idx] = if expr_has_space { 1 } else { self.default_modifier[space_idx] };
        normalized
    }

    fn apply_contextual_base_cap(&self, modifier_values: &[u16], raw_ids: &[u32], start_idx: usize, consumed_len: usize) -> Vec<u16> {
        let Some(cap_idx) = self.group_idx("base_capitalization") else {
            return modifier_values.to_vec();
        };
        let mut normalized = modifier_values.to_vec();
        if normalized[cap_idx] == self.default_modifier[cap_idx] {
            return normalized;
        }
        if self.raw_span_has_byte_fallback(raw_ids, start_idx, consumed_len) {
            normalized[cap_idx] = self.default_modifier[cap_idx];
            return normalized;
        }
        let raw_surface = self.decode_ids(&raw_ids[start_idx..usize::min(start_idx + consumed_len, raw_ids.len())]);
        if !is_capitalized_surface(&raw_surface) || !is_base_cap_representable_surface(&raw_surface) {
            normalized[cap_idx] = self.default_modifier[cap_idx];
        }
        normalized
    }

    fn entry_has_case_mismatch(&self, entry: &EntryValue, modifier_values: &[u16], raw_ids: &[u32], start_idx: usize, consumed_len: usize) -> bool {
        let Some(cap_idx) = self.group_idx("base_capitalization") else {
            return false;
        };
        if modifier_values[cap_idx] != self.default_modifier[cap_idx] {
            return false;
        }
        let raw_surface = self.decode_ids(&raw_ids[start_idx..usize::min(start_idx + consumed_len, raw_ids.len())]);
        let base_surface = self.decode_ids(&entry.base_ids);
        if raw_surface == base_surface {
            return false;
        }
        let raw_letters: String = raw_surface.chars().filter(|ch| ch.is_alphabetic()).flat_map(|ch| ch.to_lowercase()).collect();
        let base_letters: String = base_surface.chars().filter(|ch| ch.is_alphabetic()).flat_map(|ch| ch.to_lowercase()).collect();
        if raw_letters.is_empty() || raw_letters != base_letters {
            return false;
        }
        if !is_base_cap_representable_surface(&raw_surface) {
            return true;
        }
        let raw_is_upper = first_alpha_is_upper(&raw_surface);
        let base_is_upper = first_alpha_is_upper(&base_surface);
        matches!((raw_is_upper, base_is_upper), (Some(left), Some(right)) if left != right)
    }

    fn build_base_bpe_runtime(config: &BaseBpeConfig) -> PyResult<Option<BaseBpeRuntime>> {
        if config.pattern.is_empty() || config.mergeable_ranks.is_empty() {
            return Ok(None);
        }
        let mut encoder: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        let mut special_tokens: FxHashMap<String, u32> = FxHashMap::default();
        let mut max_token_id = 0usize;
        let mut id_to_token_bytes: Vec<Option<Vec<u8>>> = Vec::new();

        for entry in &config.mergeable_ranks {
            let token_bytes = latin1_to_bytes(&entry.token).ok_or_else(|| {
                PyValueError::new_err("base_bpe mergeable_ranks contains non-latin1 token text")
            })?;
            let token_id = entry.rank as usize;
            if token_id >= id_to_token_bytes.len() {
                id_to_token_bytes.resize(token_id + 1, None);
            }
            id_to_token_bytes[token_id] = Some(token_bytes.clone());
            encoder.insert(token_bytes, entry.rank);
            max_token_id = max_token_id.max(token_id);
        }

        for (token, token_id) in &config.special_tokens {
            special_tokens.insert(token.clone(), *token_id);
            let token_id_usize = *token_id as usize;
            if token_id_usize >= id_to_token_bytes.len() {
                id_to_token_bytes.resize(token_id_usize + 1, None);
            }
            id_to_token_bytes[token_id_usize] = Some(token.as_bytes().to_vec());
            max_token_id = max_token_id.max(token_id_usize);
        }

        if id_to_token_bytes.len() <= max_token_id {
            id_to_token_bytes.resize(max_token_id + 1, None);
        }

        let core = CoreBPE::new(encoder, special_tokens, &config.pattern)
            .map_err(|e| PyValueError::new_err(format!("Failed to build base_bpe runtime: {e}")))?;
        Ok(Some(BaseBpeRuntime {
            core,
            id_to_token_bytes,
        }))
    }

    fn build_token_meta_table_from_tokenizer(tokenizer: &Tokenizer, runtime: &RuntimeConfig, explicit: &[TokenMeta]) -> Vec<TokenMeta> {
        if !explicit.is_empty() {
            return explicit.to_vec();
        }
        let vocab_size = tokenizer.get_vocab_size(true);
        let det_map = runtime.literal_maps.get("determiners");
        let prep_map = runtime.literal_maps.get("prepositions");
        let prefix_map = runtime.literal_maps.get("prefix_punctuation");
        let suffix_map = runtime.literal_maps.get("suffix_punctuation");
        (0..vocab_size)
            .map(|token_id| {
                let vocab_token = tokenizer.id_to_token(token_id as u32).unwrap_or_default();
                let decoded_token = tokenizer.decode(&[token_id as u32], false).unwrap_or_default();
                let surface_text = if decoded_token.is_empty() {
                    vocab_token.as_str()
                } else {
                    decoded_token.as_str()
                };
                let canonical_surface = surface_text
                    .trim_start_matches(|ch| ch == ' ' || ch == 'Ġ' || ch == '▁')
                    .to_lowercase();
                let stripped = surface_text.trim_start_matches(|ch| ch == ' ' || ch == 'Ġ' || ch == '▁');
                let has_space_prefix = vocab_token.starts_with(' ')
                    || vocab_token.starts_with('Ġ')
                    || vocab_token.starts_with('▁')
                    || surface_text.starts_with(' ');
                TokenMeta {
                    token_text: surface_text.to_string(),
                    canonical_surface: canonical_surface.clone(),
                    has_space_prefix,
                    has_word_char: stripped.chars().any(|ch| ch.is_alphanumeric()),
                    is_whitespace_only: surface_text.trim().is_empty(),
                    is_single_ascii_space: surface_text == " ",
                    is_byte_fallback: false,
                    is_base_cap_representable: is_base_cap_representable_surface(surface_text),
                    determiner: det_map.and_then(|m| m.get(&canonical_surface)).cloned(),
                    preposition: prep_map.and_then(|m| m.get(&canonical_surface)).cloned(),
                    prefix_punctuation: prefix_map.and_then(|m| m.get(&canonical_surface)).cloned(),
                    suffix_punctuation: suffix_map.and_then(|m| m.get(&canonical_surface)).cloned(),
                }
            })
            .collect()
    }

    fn build_token_meta_table_from_base_bpe(
        base_bpe: &BaseBpeRuntime,
        runtime: &RuntimeConfig,
        explicit: &[TokenMeta],
    ) -> Vec<TokenMeta> {
        if !explicit.is_empty() {
            return explicit.to_vec();
        }
        let det_map = runtime.literal_maps.get("determiners");
        let prep_map = runtime.literal_maps.get("prepositions");
        let prefix_map = runtime.literal_maps.get("prefix_punctuation");
        let suffix_map = runtime.literal_maps.get("suffix_punctuation");

        base_bpe
            .id_to_token_bytes
            .iter()
            .enumerate()
            .map(|(token_id, maybe_vocab_bytes)| {
                let vocab_bytes = maybe_vocab_bytes.clone().unwrap_or_default();
                let is_byte_fallback = !vocab_bytes.is_empty() && std::str::from_utf8(&vocab_bytes).is_err();
                let vocab_token = bytes_to_latin1(&vocab_bytes);
                let decoded_token = if let Ok(text) = base_bpe.core.decode(&[token_id as u32]) {
                    text
                } else if let Ok(bytes) = base_bpe.core.decode_bytes(&[token_id as u32]) {
                    decode_utf8_lossy(&bytes)
                } else {
                    String::new()
                };
                let surface_text = if decoded_token.is_empty() {
                    vocab_token.as_str()
                } else {
                    decoded_token.as_str()
                };
                let canonical_surface = if is_byte_fallback {
                    String::new()
                } else {
                    surface_text
                        .trim_start_matches(|ch| ch == ' ' || ch == 'Ġ' || ch == '▁')
                        .to_lowercase()
                };
                let stripped = surface_text.trim_start_matches(|ch| ch == ' ' || ch == 'Ġ' || ch == '▁');
                let has_space_prefix = vocab_bytes.first().copied() == Some(b' ') || surface_text.starts_with(' ');
                TokenMeta {
                    token_text: surface_text.to_string(),
                    canonical_surface: canonical_surface.clone(),
                    has_space_prefix,
                    has_word_char: !is_byte_fallback && stripped.chars().any(|ch| ch.is_alphanumeric()),
                    is_whitespace_only: surface_text.trim().is_empty(),
                    is_single_ascii_space: vocab_bytes == [b' '] || surface_text == " ",
                    is_byte_fallback,
                    is_base_cap_representable: !is_byte_fallback && is_base_cap_representable_surface(surface_text),
                    determiner: if is_byte_fallback { None } else { det_map.and_then(|m| m.get(&canonical_surface)).cloned() },
                    preposition: if is_byte_fallback { None } else { prep_map.and_then(|m| m.get(&canonical_surface)).cloned() },
                    prefix_punctuation: if is_byte_fallback { None } else { prefix_map.and_then(|m| m.get(&canonical_surface)).cloned() },
                    suffix_punctuation: if is_byte_fallback { None } else { suffix_map.and_then(|m| m.get(&canonical_surface)).cloned() },
                }
            })
            .collect()
    }

    fn process_ids_impl(&self, raw_ids: &[u32]) -> (Vec<u32>, Vec<Vec<u16>>) {
        let mut out_ids = Vec::new();
        let mut out_mods = Vec::new();
        let space_prefix_prefix_sum: Vec<usize> = {
            let mut out = vec![0usize; raw_ids.len() + 1];
            let mut running = 0usize;
            for (idx, token_id) in raw_ids.iter().enumerate() {
                if self.token_meta_ref(*token_id).has_space_prefix {
                    running += 1;
                }
                out[idx + 1] = running;
            }
            out
        };

        let article_cap_idx = self.group_idx("article_capitalization");
        let prep_cap_idx = self.group_idx("prep_capitalization");
        let space_idx = self.group_idx("space_prefix");
        let suffix_group_name = self
            .group_idx("suffix_punctuation")
            .map(|_| "suffix_punctuation".to_string());

        let mut pending_groups: Vec<PendingGroup> = Vec::new();
        let mut pending_token_records: Vec<(usize, u32)> = Vec::new();
        let mut pending_leading_space = false;

        let literal_modifier = |this: &Self, start_idx: usize, token_id: u32, force_leading_space: bool| -> Vec<u16> {
            let mut modifier = this.empty_modifier();
            if let Some(idx) = space_idx {
                if !this.token_meta_ref(token_id).is_whitespace_only
                    && (force_leading_space || this.raw_expr_has_leading_space(raw_ids, start_idx))
                {
                    modifier[idx] = 1;
                }
            }
            modifier
        };

        let emit_literal = |this: &Self, start_idx: usize, token_id: u32, force_leading_space: bool, out_ids: &mut Vec<u32>, out_mods: &mut Vec<Vec<u16>>| {
            out_ids.push(token_id);
            out_mods.push(literal_modifier(this, start_idx, token_id, force_leading_space));
        };

        let flush_pending_literal = |this: &Self,
                                         pending_leading_space_ref: &mut bool,
                                         pending_groups_ref: &mut Vec<PendingGroup>,
                                         pending_token_records_ref: &mut Vec<(usize, u32)>,
                                         out_ids_ref: &mut Vec<u32>,
                                         out_mods_ref: &mut Vec<Vec<u16>>| {
            let mut emit_leading_space = *pending_leading_space_ref;
            for (raw_idx, raw_token_id) in pending_token_records_ref.iter().copied() {
                let is_whitespace_only = this.token_meta_ref(raw_token_id).is_whitespace_only;
                emit_literal(
                    this,
                    raw_idx,
                    raw_token_id,
                    emit_leading_space && !is_whitespace_only,
                    out_ids_ref,
                    out_mods_ref,
                );
                if emit_leading_space {
                    emit_leading_space = false;
                }
            }
            pending_groups_ref.clear();
            pending_token_records_ref.clear();
            *pending_leading_space_ref = false;
        };

        let mark_pending_detached_prefix = |this: &Self, start_idx: usize, pending_leading_space_ref: &mut bool| {
            if !*pending_leading_space_ref && this.raw_expr_has_leading_space(raw_ids, start_idx) {
                *pending_leading_space_ref = true;
            }
        };

        let mut idx = 0usize;
        while idx < raw_ids.len() {
            let token_id = raw_ids[idx];
            let meta = self.token_meta_ref(token_id).clone();

            if meta.is_whitespace_only {
                if meta.token_text == " " {
                    if !pending_groups.is_empty() {
                        pending_token_records.push((idx, token_id));
                    } else if self.token_can_host_expr_space(raw_ids, idx + 1) {
                        pending_leading_space = true;
                    } else {
                        emit_literal(self, idx, token_id, false, &mut out_ids, &mut out_mods);
                    }
                    idx += 1;
                    continue;
                }
                if !pending_groups.is_empty() || !pending_token_records.is_empty() {
                    flush_pending_literal(self, &mut pending_leading_space, &mut pending_groups, &mut pending_token_records, &mut out_ids, &mut out_mods);
                }
                emit_literal(self, idx, token_id, false, &mut out_ids, &mut out_mods);
                idx += 1;
                continue;
            }

            if meta.is_byte_fallback {
                let mut first_modifier = if !pending_groups.is_empty() || !pending_token_records.is_empty() {
                    self.combine_pending(&self.empty_modifier(), &pending_groups)
                } else {
                    self.empty_modifier()
                };
                if let Some(space_group_idx) = space_idx {
                    if pending_leading_space {
                        first_modifier[space_group_idx] = 1;
                    }
                }
                let component_end = self.byte_component_end(raw_ids, idx);
                for component_idx in idx..component_end {
                    out_ids.push(raw_ids[component_idx]);
                    if component_idx == idx {
                        out_mods.push(first_modifier.clone());
                    } else {
                        out_mods.push(self.empty_modifier());
                    }
                }
                pending_groups.clear();
                pending_token_records.clear();
                pending_leading_space = false;
                idx = component_end;
                continue;
            }

            if let Some(suffix_transform) = meta.suffix_punctuation.clone() {
                if self.group_idx("suffix_punctuation").is_some() && !out_mods.is_empty() {
                    let prev_is_whitespace = idx == 0 || self.token_meta_ref(raw_ids[idx - 1]).is_whitespace_only;
                    let already_has_suffix = suffix_group_name
                        .as_ref()
                        .map(|name| self.modifier_has_active_group(out_mods.last().unwrap(), name))
                        .unwrap_or(false);
                    if !prev_is_whitespace
                        && !meta.has_space_prefix
                        && !already_has_suffix
                        && self.runtime.attachment_limits.max_suffix_punctuation > 0
                    {
                        if let Some(group_idx) = self.group_idx(&suffix_transform.group_name) {
                            out_mods.last_mut().unwrap()[group_idx] = suffix_transform.rel_idx as u16;
                            idx += 1;
                            continue;
                        }
                    }
                }
            }

            if let Some(prefix_transform) = meta.prefix_punctuation.clone() {
                let next_idx = idx + 1;
                if self.runtime.attachment_limits.max_prefix_punctuation > 0
                    && next_idx < raw_ids.len()
                    && !self.token_meta_ref(raw_ids[next_idx]).is_whitespace_only
                    && !self.token_meta_ref(raw_ids[next_idx]).has_space_prefix
                    && self.raw_position_has_word_char(raw_ids, next_idx)
                {
                    if let Some(group_idx) = self.group_idx(&prefix_transform.group_name) {
                        pending_groups.push(PendingGroup {
                            group_idx,
                            rel_idx: prefix_transform.rel_idx as u16,
                        });
                        pending_token_records.push((idx, token_id));
                        idx += 1;
                        continue;
                    }
                }
                if !pending_groups.is_empty() || !pending_token_records.is_empty() {
                    flush_pending_literal(self, &mut pending_leading_space, &mut pending_groups, &mut pending_token_records, &mut out_ids, &mut out_mods);
                }
            }

            if let Some(det_transform) = meta.determiner.clone() {
                if meta.is_base_cap_representable
                    && self.can_attach_detached_modifier(raw_ids, idx, 1, !pending_groups.is_empty())
                {
                    mark_pending_detached_prefix(self, idx, &mut pending_leading_space);
                    if let Some(group_idx) = self.group_idx(&det_transform.group_name) {
                        pending_groups.push(PendingGroup {
                            group_idx,
                            rel_idx: det_transform.rel_idx as u16,
                        });
                        pending_token_records.push((idx, token_id));
                        if is_capitalized_surface(&meta.token_text) && meta.is_base_cap_representable {
                            if let Some(cap_idx) = article_cap_idx {
                                pending_groups.push(PendingGroup { group_idx: cap_idx, rel_idx: 1 });
                            }
                        }
                        idx += 1;
                        continue;
                    }
                }
            }

            if let Some(prep_transform) = meta.preposition.clone() {
                if meta.is_base_cap_representable
                    && self.can_attach_detached_modifier(raw_ids, idx, 1, !pending_groups.is_empty())
                {
                    mark_pending_detached_prefix(self, idx, &mut pending_leading_space);
                    if let Some(group_idx) = self.group_idx(&prep_transform.group_name) {
                        pending_groups.push(PendingGroup {
                            group_idx,
                            rel_idx: prep_transform.rel_idx as u16,
                        });
                        pending_token_records.push((idx, token_id));
                        if is_capitalized_surface(&meta.token_text) && meta.is_base_cap_representable {
                            if let Some(cap_idx) = prep_cap_idx {
                                pending_groups.push(PendingGroup { group_idx: cap_idx, rel_idx: 1 });
                            }
                        }
                        idx += 1;
                        continue;
                    }
                }
            }

            if !pending_groups.is_empty() && (meta.determiner.is_some() || meta.preposition.is_some()) {
                flush_pending_literal(self, &mut pending_leading_space, &mut pending_groups, &mut pending_token_records, &mut out_ids, &mut out_mods);
                continue;
            }

            let mut entry = self.find_longest_boundary_safe_match(raw_ids, idx, &space_prefix_prefix_sum);
            if let Some(found) = entry.clone() {
                if self.should_prefer_cap_fallback_over_match(raw_ids, idx, &found) {
                    entry = None;
                } else {
                    let mut combined_modifier = self.combine_modifier_rows(&found.modifier_rows);
                    let base_surface = self.decode_ids(&found.base_ids).trim_start_matches(' ').to_lowercase();
                    let raw_surface = self.decode_ids(&raw_ids[idx..idx + found.consumed_len]);

                    if found.base_ids.len() == 1
                        && self.modifier_has_only_surface_groups(&combined_modifier)
                        && self.runtime.literal_maps.get("determiners").map(|m| m.contains_key(&base_surface)).unwrap_or(false)
                        && is_base_cap_representable_surface(&raw_surface)
                        && self.can_attach_detached_modifier(raw_ids, idx, found.consumed_len, !pending_groups.is_empty())
                    {
                        mark_pending_detached_prefix(self, idx, &mut pending_leading_space);
                        if let Some(transform) = self.runtime.literal_maps.get("determiners").and_then(|m| m.get(&base_surface)) {
                            if let Some(group_idx) = self.group_idx(&transform.group_name) {
                                pending_groups.push(PendingGroup { group_idx, rel_idx: transform.rel_idx as u16 });
                                for offset in 0..found.consumed_len {
                                    pending_token_records.push((idx + offset, raw_ids[idx + offset]));
                                }
                                if is_capitalized_surface(&raw_surface) && is_base_cap_representable_surface(&raw_surface) {
                                    if let Some(cap_idx) = article_cap_idx {
                                        pending_groups.push(PendingGroup { group_idx: cap_idx, rel_idx: 1 });
                                    }
                                }
                                idx += found.consumed_len;
                                continue;
                            }
                        }
                    }

                    if found.base_ids.len() == 1
                        && self.modifier_has_only_surface_groups(&combined_modifier)
                        && self.runtime.literal_maps.get("prepositions").map(|m| m.contains_key(&base_surface)).unwrap_or(false)
                        && is_base_cap_representable_surface(&raw_surface)
                        && self.can_attach_detached_modifier(raw_ids, idx, found.consumed_len, !pending_groups.is_empty())
                    {
                        mark_pending_detached_prefix(self, idx, &mut pending_leading_space);
                        if let Some(transform) = self.runtime.literal_maps.get("prepositions").and_then(|m| m.get(&base_surface)) {
                            if let Some(group_idx) = self.group_idx(&transform.group_name) {
                                pending_groups.push(PendingGroup { group_idx, rel_idx: transform.rel_idx as u16 });
                                for offset in 0..found.consumed_len {
                                    pending_token_records.push((idx + offset, raw_ids[idx + offset]));
                                }
                                if is_capitalized_surface(&raw_surface) && is_base_cap_representable_surface(&raw_surface) {
                                    if let Some(cap_idx) = prep_cap_idx {
                                        pending_groups.push(PendingGroup { group_idx: cap_idx, rel_idx: 1 });
                                    }
                                }
                                idx += found.consumed_len;
                                continue;
                            }
                        }
                    }

                    combined_modifier = self.strip_invalid_detached_modifier_groups(
                        &combined_modifier,
                        raw_ids,
                        idx,
                        found.consumed_len,
                        !pending_groups.is_empty(),
                    );
                    combined_modifier = self.apply_contextual_base_cap(&combined_modifier, raw_ids, idx, found.consumed_len);
                    combined_modifier = self.apply_contextual_space_prefix(
                        &combined_modifier,
                        raw_ids,
                        idx,
                        !pending_groups.is_empty() || pending_leading_space,
                        pending_leading_space,
                    );
                    if self.entry_has_case_mismatch(&found, &combined_modifier, raw_ids, idx, found.consumed_len) {
                        entry = None;
                    } else {
                        let combined_modifier = self.strip_nonlexical_surface_groups(&combined_modifier, &found.base_ids);
                        if !pending_groups.is_empty() || pending_leading_space {
                            if found.base_ids.len() == 1 {
                                let mut merged = found.modifier_rows[0].clone();
                                merged = self.strip_invalid_detached_modifier_groups(
                                    &merged,
                                    raw_ids,
                                    idx,
                                    found.consumed_len,
                                    !pending_groups.is_empty(),
                                );
                                merged = self.apply_contextual_base_cap(&merged, raw_ids, idx, found.consumed_len);
                                merged = self.apply_contextual_space_prefix(
                                    &merged,
                                    raw_ids,
                                    idx,
                                    !pending_groups.is_empty() || pending_leading_space,
                                    pending_leading_space,
                                );
                                merged = self.strip_nonlexical_surface_groups(&merged, &found.base_ids);
                                merged = self.combine_pending(&merged, &pending_groups);
                                out_ids.extend(found.base_ids.iter().copied());
                                out_mods.push(merged);
                            } else {
                                let mut combined = combined_modifier.clone();
                                if let Some(space_group_idx) = space_idx {
                                    if pending_leading_space {
                                        combined[space_group_idx] = 1;
                                    }
                                }
                                combined = self.combine_pending(&combined, &pending_groups);
                                out_ids.extend(found.base_ids.iter().copied());
                                out_mods.extend(self.spread_multi_token_modifiers(&combined, found.base_ids.len()));
                            }
                        } else {
                            out_ids.extend(found.base_ids.iter().copied());
                            if found.base_ids.len() == 1 {
                                out_mods.push(combined_modifier);
                            } else {
                                let mut normalized_rows = found.modifier_rows.clone();
                                if !normalized_rows.is_empty() {
                                    normalized_rows[0] = self.apply_contextual_base_cap(&normalized_rows[0], raw_ids, idx, found.consumed_len);
                                    normalized_rows[0] = self.apply_contextual_space_prefix(
                                        &normalized_rows[0],
                                        raw_ids,
                                        idx,
                                        pending_leading_space,
                                        pending_leading_space,
                                    );
                                    if let Some(space_group_idx) = space_idx {
                                        for row in normalized_rows.iter_mut().skip(1) {
                                            row[space_group_idx] = self.default_modifier[space_group_idx];
                                        }
                                    }
                                }
                                out_mods.extend(normalized_rows);
                            }
                        }
                        pending_groups.clear();
                        pending_token_records.clear();
                        pending_leading_space = false;
                        idx += found.consumed_len;
                        continue;
                    }
                }
            }

            if meta.has_word_char {
                if let Some((consumed_len, fallback_ids, mut fallback_mods)) =
                    self.try_lowercase_cap_fallback(raw_ids, idx, &pending_groups, pending_leading_space)
                {
                    let use_pending_space = if !pending_groups.is_empty() { true } else { pending_leading_space };
                    fallback_mods[0] = self.apply_contextual_space_prefix(
                        &fallback_mods[0],
                        raw_ids,
                        idx,
                        use_pending_space,
                        pending_leading_space,
                    );
                    out_ids.extend(fallback_ids);
                    out_mods.extend(fallback_mods);
                    pending_groups.clear();
                    pending_token_records.clear();
                    pending_leading_space = false;
                    idx += consumed_len;
                    continue;
                }
            }

            let mut base_modifier = self.empty_modifier();
            if let Some(space_group_idx) = space_idx {
                if pending_leading_space && self.raw_position_has_word_char(raw_ids, idx) {
                    base_modifier[space_group_idx] = 1;
                }
            }
            base_modifier = self.apply_contextual_base_cap(&base_modifier, raw_ids, idx, 1);
            base_modifier = self.apply_contextual_space_prefix(
                &base_modifier,
                raw_ids,
                idx,
                pending_leading_space,
                pending_leading_space,
            );
            base_modifier = self.combine_pending(&base_modifier, &pending_groups);
            out_ids.push(token_id);
            out_mods.push(base_modifier);
            pending_groups.clear();
            pending_token_records.clear();
            pending_leading_space = false;
            idx += 1;
        }

        if !pending_groups.is_empty() || !pending_token_records.is_empty() {
            flush_pending_literal(self, &mut pending_leading_space, &mut pending_groups, &mut pending_token_records, &mut out_ids, &mut out_mods);
        }
        (out_ids, out_mods)
    }

    fn lexical_surface_for_reverse_entry(&self, entry: &ReverseEntryValue) -> String {
        if entry.base_ids.len() == 1 {
            return self.decode_ids(&entry.base_ids);
        }
        if let Some(surface) = entry.surface.as_ref() {
            return surface.clone();
        }
        self.decode_ids(&entry.base_ids)
    }

    fn reconstruct_surface_impl(&self, token_ids: &[u32], modifier_rows: &[Vec<u16>]) -> PyResult<String> {
        if token_ids.len() != modifier_rows.len() {
            return Err(PyValueError::new_err(format!(
                "token_ids and modifier_rows length mismatch: {} != {}",
                token_ids.len(),
                modifier_rows.len()
            )));
        }
        let mut chunks = Vec::new();
        let mut idx = 0usize;
        while idx < token_ids.len() {
            if let Some(entry) = self.longest_reverse_match(token_ids, modifier_rows, idx) {
                let lexical_surface = self.lexical_surface_for_reverse_entry(&entry);
                let combined = self.combine_modifier_rows(&entry.modifier_rows);
                chunks.push(self.synthesize_surface(&lexical_surface, &combined));
                idx += entry.consumed_len;
                continue;
            }
            if self.token_meta_ref(token_ids[idx]).is_byte_fallback {
                let component_end = self.byte_component_end(token_ids, idx);
                if let Some(lexical_surface) = self.decode_token_bytes(&token_ids[idx..component_end]) {
                    let combined = self.combine_modifier_rows(&modifier_rows[idx..component_end]);
                    if combined == self.default_modifier {
                        chunks.push(lexical_surface);
                    } else {
                        chunks.push(self.synthesize_surface(&lexical_surface, &combined));
                    }
                    idx = component_end;
                    continue;
                }
            }
            if modifier_rows[idx] == self.default_modifier {
                let literal_start = idx;
                idx += 1;
                while idx < token_ids.len() {
                    if modifier_rows[idx] != self.default_modifier {
                        break;
                    }
                    if self.token_meta_ref(token_ids[idx]).is_byte_fallback {
                        break;
                    }
                    if self.longest_reverse_match(token_ids, modifier_rows, idx).is_some() {
                        break;
                    }
                    idx += 1;
                }
                chunks.push(self.decode_ids(&token_ids[literal_start..idx]));
                continue;
            }
            let lexical_surface = self.decode_single(token_ids[idx]);
            chunks.push(self.synthesize_surface(&lexical_surface, &modifier_rows[idx]));
            idx += 1;
        }
        Ok(chunks.concat())
    }

    fn encode_text_impl(&self, text: &str) -> PyResult<Vec<u32>> {
        if let Some(runtime) = self.base_bpe.as_ref() {
            return Ok(runtime.core.encode_ordinary(text));
        }
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Rust compositional processor has no text encoder configured."))?;
        let encoding = tokenizer
            .encode(text, false)
            .map_err(|e| PyValueError::new_err(format!("Failed to encode text in Rust compositional processor: {e}")))?;
        Ok(encoding.get_ids().iter().map(|v| *v as u32).collect())
    }

    fn build_result_object(&self, py: Python<'_>, output_ids: Vec<u32>, modifier_rows: Vec<Vec<u16>>) -> PyObject {
        (output_ids, modifier_rows).into_py(py)
    }

    fn group_value_name(&self, group_name: &str, rel_idx: usize) -> Option<String> {
        self.group_value_names
            .get(group_name)
            .and_then(|values| values.get(rel_idx))
            .cloned()
    }

    fn literal_debug_value(&self, literal: &Option<LiteralTransform>) -> serde_json::Value {
        let Some(transform) = literal else {
            return serde_json::Value::Null;
        };
        json!({
            "group_name": transform.group_name,
            "rel_idx": transform.rel_idx,
            "value_name": self.group_value_name(&transform.group_name, transform.rel_idx),
        })
    }

    fn token_debug_value(&self, token_id: u32) -> serde_json::Value {
        let meta = self.token_meta_ref(token_id);
        let vocab_token = self.base_bpe_vocab_token(token_id).or_else(|| {
            self.tokenizer
                .as_ref()
                .and_then(|tok| tok.id_to_token(token_id))
        });
        let decoded_token = self.base_bpe_decoded_token(token_id).or_else(|| {
            self.tokenizer
                .as_ref()
                .and_then(|tok| tok.decode(&[token_id], false).ok())
        });
        json!({
            "id": token_id,
            "vocab_token": vocab_token,
            "decoded_token": decoded_token,
            "token_text": meta.token_text,
            "canonical_surface": meta.canonical_surface,
            "has_space_prefix": meta.has_space_prefix,
            "has_word_char": meta.has_word_char,
            "is_whitespace_only": meta.is_whitespace_only,
            "is_single_ascii_space": meta.is_single_ascii_space,
            "is_byte_fallback": meta.is_byte_fallback,
            "is_base_cap_representable": meta.is_base_cap_representable,
            "determiner": self.literal_debug_value(&meta.determiner),
            "preposition": self.literal_debug_value(&meta.preposition),
            "prefix_punctuation": self.literal_debug_value(&meta.prefix_punctuation),
            "suffix_punctuation": self.literal_debug_value(&meta.suffix_punctuation),
        })
    }
}

#[pymethods]
impl CompositionalProcessor {
    #[new]
    fn new(config_json: &str) -> PyResult<Self> {
        let cfg: Config = serde_json::from_str(config_json)
            .map_err(|e| PyValueError::new_err(format!("Failed to parse compositional runtime config: {e}")))?;
        if cfg.version != 1 {
            return Err(PyValueError::new_err(format!(
                "Unsupported compositional runtime config version: {}",
                cfg.version
            )));
        }
        if cfg.default_modifier.len() != cfg.num_modifier_groups {
            return Err(PyValueError::new_err("default_modifier width must match num_modifier_groups"));
        }
        let tokenizer = match cfg.tokenizer_json.as_ref() {
            Some(raw_json) if !raw_json.is_empty() => {
                Some(
                    Tokenizer::from_bytes(raw_json.as_bytes())
                        .map_err(|e| PyValueError::new_err(format!("Failed to parse tokenizer_json: {e}")))?,
                )
            }
            _ => None,
        };

        let base_bpe = match cfg.base_bpe.as_ref() {
            Some(config) => Self::build_base_bpe_runtime(config)?,
            None => None,
        };

        let mut token_meta = if let Some(runtime) = base_bpe.as_ref() {
            Self::build_token_meta_table_from_base_bpe(runtime, &cfg.runtime, &cfg.token_meta)
        } else {
            match tokenizer.as_ref() {
                Some(tok) => Self::build_token_meta_table_from_tokenizer(tok, &cfg.runtime, &cfg.token_meta),
                None if !cfg.token_meta.is_empty() => cfg.token_meta.clone(),
                None => {
                    return Err(PyValueError::new_err(
                        "Rust compositional processor requires base_bpe, tokenizer_json, or populated token_meta.",
                    ))
                }
            }
        };
        if token_meta.is_empty() {
            token_meta.push(TokenMeta::default());
        }

        let mut processor = Self {
            base_bpe,
            tokenizer,
            trie_nodes: vec![TrieNode::default()],
            max_sequence_len: 1,
            reverse_trie_nodes: vec![ReverseTrieNode::default()],
            max_reverse_sequence_len: 1,
            default_modifier: cfg.default_modifier.clone(),
            num_modifier_groups: cfg.num_modifier_groups,
            group_names: cfg.group_names.clone(),
            group_value_names: cfg.group_value_names.clone(),
            runtime: cfg.runtime.clone(),
            token_meta,
        };

        for entry in &cfg.entries {
            if entry.modifier_rows.len() != entry.base_ids.len() {
                return Err(PyValueError::new_err("modifier_rows length must match base_ids length"));
            }
            for row in &entry.modifier_rows {
                if row.len() != processor.num_modifier_groups {
                    return Err(PyValueError::new_err("modifier row width mismatch"));
                }
            }
            processor.add_entry(entry);
        }
        for entry in &cfg.reverse_entries {
            if entry.modifier_rows.len() != entry.base_ids.len() {
                return Err(PyValueError::new_err("reverse entry modifier_rows length must match base_ids length"));
            }
            for row in &entry.modifier_rows {
                if row.len() != processor.num_modifier_groups {
                    return Err(PyValueError::new_err("reverse entry modifier row width mismatch"));
                }
            }
            if entry.token_ids.is_empty() || entry.base_ids.is_empty() {
                return Err(PyValueError::new_err("reverse entries must provide non-empty token_ids and base_ids"));
            }
            processor.add_reverse_entry(entry);
        }
        Ok(processor)
    }

    fn process_ids(&self, raw_ids: Vec<u32>, py: Python<'_>) -> PyResult<PyObject> {
        let (output_ids, modifier_rows) = self.process_ids_impl(&raw_ids);
        Ok(self.build_result_object(py, output_ids, modifier_rows))
    }

    fn process_ids_batch(&self, raw_ids_batch: Vec<Vec<u32>>, py: Python<'_>) -> PyResult<PyObject> {
        let out = PyList::empty_bound(py);
        for raw_ids in raw_ids_batch {
            let (output_ids, modifier_rows) = self.process_ids_impl(&raw_ids);
            out.append(self.build_result_object(py, output_ids, modifier_rows))?;
        }
        Ok(out.into_py(py))
    }

    fn process_text(&self, text: String, py: Python<'_>) -> PyResult<PyObject> {
        let raw_ids = self.encode_text_impl(text.as_str())?;
        let (output_ids, modifier_rows) = self.process_ids_impl(&raw_ids);
        Ok(self.build_result_object(py, output_ids, modifier_rows))
    }

    fn process_text_batch(&self, texts: Vec<String>, py: Python<'_>) -> PyResult<PyObject> {
        let out = PyList::empty_bound(py);
        if let Some(runtime) = self.base_bpe.as_ref() {
            let raw_ids_batch: Vec<Vec<u32>> = texts
                .iter()
                .map(|text| runtime.core.encode_ordinary(text))
                .collect();
            let processed: Vec<(Vec<u32>, Vec<Vec<u16>>)> = py.allow_threads(|| {
                raw_ids_batch
                    .par_iter()
                    .map(|raw_ids| self.process_ids_impl(raw_ids))
                    .collect()
            });
            for (output_ids, modifier_rows) in processed {
                out.append(self.build_result_object(py, output_ids, modifier_rows))?;
            }
            return Ok(out.into_py(py));
        }
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Rust compositional processor has no text encoder configured."))?;
        let input_refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let encodings = tokenizer
            .encode_batch(input_refs, false)
            .map_err(|e| PyValueError::new_err(format!("Failed to encode batch in Rust compositional processor: {e}")))?;
        let raw_ids_batch: Vec<Vec<u32>> = encodings
            .into_iter()
            .map(|encoding| encoding.get_ids().iter().map(|v| *v as u32).collect())
            .collect();
        let processed: Vec<(Vec<u32>, Vec<Vec<u16>>)> = py.allow_threads(|| {
            raw_ids_batch
                .par_iter()
                .map(|raw_ids| self.process_ids_impl(raw_ids))
                .collect()
        });
        for (output_ids, modifier_rows) in processed {
            out.append(self.build_result_object(py, output_ids, modifier_rows))?;
        }
        Ok(out.into_py(py))
    }

    fn decode_with_modifiers(&self, token_ids: Vec<u32>, modifier_rows: Vec<Vec<u16>>) -> PyResult<String> {
        self.reconstruct_surface_impl(&token_ids, &modifier_rows)
    }

    fn utf8_len_with_modifiers_batch(&self, token_ids: Vec<u32>, modifier_rows: Vec<Vec<u16>>) -> PyResult<Vec<u32>> {
        if token_ids.len() != modifier_rows.len() {
            return Err(PyValueError::new_err(format!(
                "token_ids and modifier_rows length mismatch: {} != {}",
                token_ids.len(),
                modifier_rows.len()
            )));
        }
        let mut out = Vec::with_capacity(token_ids.len());
        for (token_id, modifier_row) in token_ids.into_iter().zip(modifier_rows.into_iter()) {
            if modifier_row.len() != self.num_modifier_groups {
                return Err(PyValueError::new_err(format!(
                    "modifier row width mismatch: expected {}, got {}",
                    self.num_modifier_groups,
                    modifier_row.len()
                )));
            }
            if self.token_meta_ref(token_id).is_byte_fallback {
                if let (Some(token_bytes), Some(delta)) = (
                    self.base_bpe_token_bytes(token_id),
                    self.modifier_utf8_delta(&modifier_row),
                ) {
                    out.push((token_bytes.len() + delta) as u32);
                    continue;
                }
            }
            let surface = self.reconstruct_surface_impl(&[token_id], &[modifier_row])?;
            out.push(surface.as_bytes().len() as u32);
        }
        Ok(out)
    }

    fn debug_tokenize_text_json(&self, text: String) -> PyResult<String> {
        let raw_ids = self.encode_text_impl(&text)?;
        let raw_tokens: Vec<serde_json::Value> = raw_ids
            .iter()
            .map(|token_id| self.token_debug_value(*token_id))
            .collect();
        Ok(json!({
            "text": text,
            "raw_ids": raw_ids,
            "raw_tokens": raw_tokens,
        })
        .to_string())
    }

    fn debug_process_text_json(&self, text: String) -> PyResult<String> {
        let raw_ids = self.encode_text_impl(&text)?;
        let raw_tokens: Vec<serde_json::Value> = raw_ids
            .iter()
            .map(|token_id| self.token_debug_value(*token_id))
            .collect();
        let (output_ids, modifier_rows) = self.process_ids_impl(&raw_ids);
        let roundtrip_text = self.reconstruct_surface_impl(&output_ids, &modifier_rows)?;
        Ok(json!({
            "text": text,
            "raw_ids": raw_ids,
            "raw_tokens": raw_tokens,
            "output_ids": output_ids,
            "modifier_rows": modifier_rows,
            "roundtrip_text": roundtrip_text,
        })
        .to_string())
    }
}

#[pymodule]
fn nanochat_compositional_rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CompositionalProcessor>()?;
    Ok(())
}
