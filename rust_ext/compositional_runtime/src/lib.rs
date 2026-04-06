use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde::Deserialize;
use std::collections::HashMap;
use tokenizers::Tokenizer;

#[derive(Debug, Deserialize, Clone)]
struct Config {
    version: usize,
    num_modifier_groups: usize,
    default_modifier: Vec<u16>,
    entries: Vec<Entry>,
    tokenizer_json: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
struct Entry {
    token_ids: Vec<u32>,
    base_ids: Vec<u32>,
    modifier_rows: Vec<Vec<u16>>,
}

#[derive(Clone)]
struct EntryValue {
    consumed_len: usize,
    base_ids: Vec<u32>,
    modifier_rows: Vec<Vec<u16>>,
}

#[derive(Default)]
struct TrieNode {
    children: HashMap<u32, usize>,
    value: Option<EntryValue>,
}

#[pyclass]
struct CompositionalProcessor {
    tokenizer: Option<Tokenizer>,
    trie_nodes: Vec<TrieNode>,
    max_sequence_len: usize,
    default_modifier: Vec<u16>,
    num_modifier_groups: usize,
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

    fn longest_match(&self, raw_ids: &[u32], start_idx: usize) -> Option<&EntryValue> {
        let mut node_idx = 0usize;
        let mut best: Option<&EntryValue> = None;
        let stop = usize::min(start_idx + self.max_sequence_len, raw_ids.len());
        for pos in start_idx..stop {
            let token_id = raw_ids[pos];
            let child_idx = match self.trie_nodes[node_idx].children.get(&token_id) {
                Some(idx) => *idx,
                None => break,
            };
            node_idx = child_idx;
            if let Some(ref value) = self.trie_nodes[node_idx].value {
                best = Some(value);
            }
        }
        best
    }

    fn process_ids_impl(&self, raw_ids: &[u32]) -> (Vec<u32>, Vec<Vec<u16>>) {
        let mut output_ids = Vec::new();
        let mut modifier_rows = Vec::new();
        let mut idx = 0usize;
        while idx < raw_ids.len() {
            if let Some(value) = self.longest_match(raw_ids, idx) {
                for token_id in &value.base_ids {
                    output_ids.push(*token_id);
                }
                for row in &value.modifier_rows {
                    modifier_rows.push(row.clone());
                }
                idx += value.consumed_len.max(1);
                continue;
            }
            output_ids.push(raw_ids[idx]);
            modifier_rows.push(self.default_modifier.clone());
            idx += 1;
        }
        (output_ids, modifier_rows)
    }

    fn encode_text_impl(&self, text: &str) -> PyResult<Vec<u32>> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Rust compositional processor has no tokenizer_json configured."))?;
        let encoding = tokenizer
            .encode(text, false)
            .map_err(|e| PyValueError::new_err(format!("Failed to encode text in Rust compositional processor: {e}")))?;
        Ok(encoding.get_ids().iter().map(|v| *v as u32).collect())
    }

    fn build_result_dict<'py>(&self, py: Python<'py>, output_ids: Vec<u32>, modifier_rows: Vec<Vec<u16>>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new_bound(py);
        out.set_item("output_ids", output_ids)?;
        let rows = PyList::empty_bound(py);
        for row in modifier_rows {
            rows.append(row)?;
        }
        out.set_item("modifier_rows", rows)?;
        Ok(out)
    }
}

#[pymethods]
impl CompositionalProcessor {
    #[new]
    fn new(config_json: &str) -> PyResult<Self> {
        let cfg: Config = serde_json::from_str(config_json)
            .map_err(|e| PyValueError::new_err(format!("Failed to parse compositional runtime config: {e}")))?;
        if cfg.version != 1 {
            return Err(PyValueError::new_err(format!("Unsupported compositional runtime config version: {}", cfg.version)));
        }
        if cfg.default_modifier.len() != cfg.num_modifier_groups {
            return Err(PyValueError::new_err("default_modifier width must match num_modifier_groups"));
        }
        let tokenizer = match cfg.tokenizer_json.as_ref() {
            Some(raw_json) if !raw_json.is_empty() => {
                let tok = Tokenizer::from_bytes(raw_json.as_bytes())
                    .map_err(|e| PyValueError::new_err(format!("Failed to parse tokenizer_json: {e}")))?;
                Some(tok)
            }
            _ => None,
        };

        let mut processor = Self {
            tokenizer,
            trie_nodes: vec![TrieNode::default()],
            max_sequence_len: 1,
            default_modifier: cfg.default_modifier.clone(),
            num_modifier_groups: cfg.num_modifier_groups,
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
        Ok(processor)
    }

    fn process_ids(&self, raw_ids: Vec<u32>, py: Python<'_>) -> PyResult<PyObject> {
        let (output_ids, modifier_rows) = self.process_ids_impl(&raw_ids);
        Ok(self.build_result_dict(py, output_ids, modifier_rows)?.into_py(py))
    }

    fn process_ids_batch(&self, raw_ids_batch: Vec<Vec<u32>>, py: Python<'_>) -> PyResult<PyObject> {
        let out = PyList::empty_bound(py);
        for raw_ids in raw_ids_batch {
            let (output_ids, modifier_rows) = self.process_ids_impl(&raw_ids);
            out.append(self.build_result_dict(py, output_ids, modifier_rows)?)?;
        }
        Ok(out.into_py(py))
    }

    fn process_text(&self, text: String, py: Python<'_>) -> PyResult<PyObject> {
        let raw_ids = self.encode_text_impl(text.as_str())?;
        let (output_ids, modifier_rows) = self.process_ids_impl(&raw_ids);
        Ok(self.build_result_dict(py, output_ids, modifier_rows)?.into_py(py))
    }

    fn process_text_batch(&self, texts: Vec<String>, py: Python<'_>) -> PyResult<PyObject> {
        let out = PyList::empty_bound(py);
        for text in texts {
            let raw_ids = self.encode_text_impl(text.as_str())?;
            let (output_ids, modifier_rows) = self.process_ids_impl(&raw_ids);
            out.append(self.build_result_dict(py, output_ids, modifier_rows)?)?;
        }
        Ok(out.into_py(py))
    }
}

#[pymodule]
fn nanochat_compositional_rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CompositionalProcessor>()?;
    Ok(())
}
