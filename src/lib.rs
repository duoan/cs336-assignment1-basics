use std::borrow::Cow;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::PathBuf;
use std::rc::Rc;

use mimalloc::MiMalloc;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
use rayon::prelude::*;
use regex::bytes::Regex;
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::LazyLock;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

static PAT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+"
    )
    .expect("failed to compile PAT regex")
});

// ─── Chunk processor (all-bytes, zero-copy for common matches) ──────────────

fn process_chunk<'a>(
    byte_data: &'a [u8],
    special_re: Option<&Regex>,
    special_set: &FxHashSet<&[u8]>,
) -> FxHashMap<Vec<u8>, usize> {
    let mut wc: FxHashMap<Cow<'a, [u8]>, usize> = FxHashMap::default();

    let parts: Vec<&'a [u8]> = match special_re {
        Some(re) => {
            let mut parts = Vec::new();
            let mut last = 0;
            for m in re.find_iter(byte_data) {
                if last < m.start() {
                    parts.push(&byte_data[last..m.start()]);
                }
                parts.push(m.as_bytes());
                last = m.end();
            }
            if last < byte_data.len() {
                parts.push(&byte_data[last..]);
            }
            parts
        }
        None => vec![byte_data],
    };

    for part in parts {
        if part.is_empty() { continue; }
        if !special_set.is_empty() && special_set.contains(part) {
            *wc.entry(Cow::Borrowed(part)).or_default() += 1;
            continue;
        }
        let mut pending_space = false;
        for m in PAT.find_iter(part) {
            let matched = m.as_bytes();

            if pending_space {
                let mut v = Vec::with_capacity(1 + matched.len());
                v.push(b' ');
                v.extend_from_slice(matched);
                *wc.entry(Cow::Owned(v)).or_default() += 1;
                pending_space = false;
                continue;
            }

            let is_all_ws = matched.len() > 1
                && matched.iter().all(|&b| b.is_ascii_whitespace());
            if is_all_ws && m.end() < part.len() && !part[m.end()].is_ascii_whitespace() {
                let last_byte = matched[matched.len() - 1];
                let trimmed = &matched[..matched.len() - 1];
                if !trimmed.is_empty() {
                    *wc.entry(Cow::Borrowed(trimmed)).or_default() += 1;
                }
                if last_byte == b' ' {
                    pending_space = true;
                } else {
                    *wc.entry(Cow::Owned(vec![last_byte])).or_default() += 1;
                }
            } else {
                *wc.entry(Cow::Borrowed(matched)).or_default() += 1;
            }
        }
    }

    wc.into_iter().map(|(k, v)| (k.into_owned(), v)).collect()
}

// ─── Core BPE logic (pure Rust, no Python types) ───────────────────────────

fn train_bpe_core(
    input_path: &PathBuf,
    vocab_size: usize,
    special_tokens: &[String],
    chunk_size: usize,
) -> (FxHashMap<usize, Vec<u8>>, Vec<(Vec<u8>, Vec<u8>)>) {
    use std::time::Instant;
    let t0 = Instant::now();

    // ── 1. Initialize token table (Rc<[u8]> for O(1) clone) ─────────────
    let mut token_table: Vec<Rc<[u8]>> = Vec::new();
    let mut token_to_id: FxHashMap<Vec<u8>, u32> = FxHashMap::default();

    for tok in special_tokens {
        let b = tok.as_bytes().to_vec();
        if !token_to_id.contains_key(&b) {
            let id = token_table.len() as u32;
            token_to_id.insert(b.clone(), id);
            token_table.push(Rc::from(b.into_boxed_slice()));
        }
    }
    for byte in 0u8..=255 {
        let v = vec![byte];
        if !token_to_id.contains_key(&v) {
            let id = token_table.len() as u32;
            token_to_id.insert(v.clone(), id);
            token_table.push(Rc::from(v.into_boxed_slice()));
        }
    }
    let initial_vocab_size = token_table.len();

    // ── 2. Streaming read + parallel pre-tokenization ────────────────────
    //
    // Read chunks in small batches (not all into memory), process in parallel,
    // merge word counts incrementally. Then single pass to build word_splits,
    // word_freqs, pair_freq, pair_to_words together.
    let special_set: FxHashSet<&[u8]> = special_tokens.iter().map(|s| s.as_bytes()).collect();
    let special_re = if !special_tokens.is_empty() {
        let alt = special_tokens
            .iter()
            .map(|t| regex::escape(t))
            .collect::<Vec<_>>()
            .join("|");
        Some(Regex::new(&format!("({})", alt)).expect("bad special-token regex"))
    } else {
        None
    };

    let n_threads = rayon::current_num_threads();
    let batch_cap = n_threads * 2;
    let mut file = BufReader::with_capacity(
        chunk_size, File::open(input_path).expect("failed to open input file"),
    );
    let mut remainder: Vec<u8> = Vec::new();
    let mut chunk_count = 0usize;

    let mut word_to_idx: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
    let mut word_splits: Vec<Vec<u32>> = Vec::new();
    let mut word_freqs: Vec<u64> = Vec::new();

    loop {
        let mut batch: Vec<Vec<u8>> = Vec::with_capacity(batch_cap);
        let mut eof = false;

        while batch.len() < batch_cap && !eof {
            let mut buf = vec![0u8; chunk_size];
            let n = file.read(&mut buf).expect("read error");
            if n == 0 {
                if !remainder.is_empty() {
                    batch.push(std::mem::take(&mut remainder));
                }
                eof = true;
            } else {
                buf.truncate(n);
                let mut combined = std::mem::take(&mut remainder);
                combined.extend_from_slice(&buf);
                match combined.iter().rposition(|&b| b == b' ' || b == b'\n') {
                    None => remainder = combined,
                    Some(pos) => {
                        remainder = combined[pos + 1..].to_vec();
                        combined.truncate(pos + 1);
                        batch.push(combined);
                    }
                }
            }
        }

        if batch.is_empty() { break; }
        chunk_count += batch.len();

        let batch_result: FxHashMap<Vec<u8>, usize> = batch
            .par_iter()
            .map(|c| process_chunk(c, special_re.as_ref(), &special_set))
            .reduce(FxHashMap::default, |mut acc, m| {
                for (k, v) in m { *acc.entry(k).or_default() += v; }
                acc
            });

        for (word, count) in batch_result {
            match word_to_idx.get(&word) {
                Some(&idx) => {
                    word_freqs[idx as usize] += count as u64;
                }
                None => {
                    let idx = word_splits.len() as u32;
                    let split: Vec<u32> = if special_set.contains(word.as_slice()) {
                        vec![token_to_id[&word]]
                    } else {
                        word.iter().map(|&b| token_to_id[&vec![b]]).collect()
                    };
                    word_splits.push(split);
                    word_freqs.push(count as u64);
                    word_to_idx.insert(word, idx);
                }
            }
        }

        if eof { break; }
    }

    drop(file);
    drop(special_re);
    drop(special_set);
    drop(token_to_id);
    drop(word_to_idx);

    eprintln!("[BPE] Pre-tokenization done: {} chunks, {} words, {:.2}s elapsed",
        chunk_count, word_splits.len(), t0.elapsed().as_secs_f64());

    // ── 3. Build pair statistics (single pass over final splits) ──────────
    let mut pair_freq: FxHashMap<(u32, u32), i64> = FxHashMap::default();
    let mut pair_to_words: FxHashMap<(u32, u32), FxHashSet<u32>> = FxHashMap::default();

    for (wid, split) in word_splits.iter().enumerate() {
        let c = word_freqs[wid] as i64;
        for w in split.windows(2) {
            let pair = (w[0], w[1]);
            *pair_freq.entry(pair).or_default() += c;
            pair_to_words.entry(pair).or_default().insert(wid as u32);
        }
    }

    eprintln!("[BPE] Pair stats built: {} pairs, {:.2}s elapsed",
        pair_freq.len(), t0.elapsed().as_secs_f64());

    // ── 5. Merge loop (BinaryHeap with lazy deletion) ───────────────────
    //
    // Heap entries: (freq, sort_key, pair_ids).
    // sort_key uses Rc<[u8]> for O(1) clone, with correct lex tiebreaking.
    // Lazy deletion: when popped freq != pair_freq[pair], entry is stale.
    let mut heap: BinaryHeap<(i64, (Rc<[u8]>, Rc<[u8]>), (u32, u32))> = BinaryHeap::new();
    for (&pair, &freq) in &pair_freq {
        heap.push((
            freq,
            (token_table[pair.0 as usize].clone(), token_table[pair.1 as usize].clone()),
            pair,
        ));
    }

    let mut merges: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    let num_merges = vocab_size.saturating_sub(initial_vocab_size);
    let log_interval = std::cmp::max(num_merges / 20, 1);

    eprintln!("[BPE] Starting merge loop: {} merges to go", num_merges);

    for _i in 0..num_merges {
        if heap.len() > pair_freq.len() * 3 {
            heap = pair_freq.iter()
                .map(|(&pair, &freq)| (
                    freq,
                    (token_table[pair.0 as usize].clone(), token_table[pair.1 as usize].clone()),
                    pair,
                ))
                .collect();
        }

        let best_pair = loop {
            match heap.pop() {
                None => break None,
                Some((freq, _sort_key, pair)) => {
                    let actual = pair_freq.get(&pair).copied().unwrap_or(0);
                    if actual == freq && freq > 0 {
                        break Some(pair);
                    }
                }
            }
        };

        let best_pair = match best_pair {
            Some(p) => p,
            None => break,
        };

        let new_bytes: Vec<u8> = [
            &*token_table[best_pair.0 as usize],
            &*token_table[best_pair.1 as usize],
        ].concat();
        let new_id = token_table.len() as u32;

        merges.push((
            token_table[best_pair.0 as usize].to_vec(),
            token_table[best_pair.1 as usize].to_vec(),
        ));

        token_table.push(Rc::from(new_bytes.into_boxed_slice()));

        let word_ids: Vec<u32> = pair_to_words
            .remove(&best_pair)
            .unwrap_or_default()
            .into_iter()
            .collect();
        pair_freq.remove(&best_pair);

        let mut dirty_pairs: FxHashSet<(u32, u32)> = FxHashSet::default();

        for &wid in &word_ids {
            let idx = wid as usize;
            let count = word_freqs[idx] as i64;
            let old_split = &word_splits[idx];

            for w in old_split.windows(2) {
                let p = (w[0], w[1]);
                if p == best_pair { continue; }
                if let Some(f) = pair_freq.get_mut(&p) {
                    *f -= count;
                    if *f <= 0 { pair_freq.remove(&p); }
                }
                if let Some(set) = pair_to_words.get_mut(&p) {
                    set.remove(&wid);
                }
                dirty_pairs.insert(p);
            }

            let mut new_split: Vec<u32> = Vec::with_capacity(old_split.len());
            let mut j = 0;
            while j < old_split.len() {
                if j + 1 < old_split.len()
                    && old_split[j] == best_pair.0
                    && old_split[j + 1] == best_pair.1
                {
                    new_split.push(new_id);
                    j += 2;
                } else {
                    new_split.push(old_split[j]);
                    j += 1;
                }
            }

            for w in new_split.windows(2) {
                let p = (w[0], w[1]);
                *pair_freq.entry(p).or_default() += count;
                pair_to_words.entry(p).or_default().insert(wid);
                dirty_pairs.insert(p);
            }

            word_splits[idx] = new_split;
        }

        for p in dirty_pairs {
            if let Some(&freq) = pair_freq.get(&p) {
                if freq > 0 {
                    heap.push((
                        freq,
                        (token_table[p.0 as usize].clone(), token_table[p.1 as usize].clone()),
                        p,
                    ));
                }
            }
        }

        if (_i + 1) % log_interval == 0 || _i + 1 == num_merges {
            eprintln!("[BPE] Merge {}/{} ({:.1}%), {:.2}s elapsed",
                _i + 1, num_merges,
                (_i + 1) as f64 / num_merges as f64 * 100.0,
                t0.elapsed().as_secs_f64());
        }
    }

    // ── 6. Build output vocab ────────────────────────────────────────────
    let mut vocab: FxHashMap<usize, Vec<u8>> = FxHashMap::default();
    let final_size = initial_vocab_size + merges.len();
    for id in 0..final_size {
        vocab.insert(id, token_table[id].to_vec());
    }

    eprintln!("[BPE] Done! {} merges, vocab size {}, total {:.2}s",
        merges.len(), vocab.len(), t0.elapsed().as_secs_f64());

    (vocab, merges)
}

// ─── PyO3 wrapper ───────────────────────────────────────────────────────────

/// Train a BPE tokenizer and return (vocab, merges).
///
/// Args:
///     input_path (str): Path to the training corpus.
///     vocab_size (int): Target vocabulary size (including special tokens).
///     special_tokens (list[str]): Tokens that are never split.
///     chunk_size (int): File read buffer in bytes (default 16 MB).
///
/// Returns:
///     tuple[dict[int, bytes], list[tuple[bytes, bytes]]]
#[pyfunction]
#[pyo3(signature = (input_path, vocab_size, special_tokens, chunk_size = 16 * 1024 * 1024))]
fn run_train_bpe<'py>(
    py: Python<'py>,
    input_path: PathBuf,
    vocab_size: usize,
    special_tokens: Vec<String>,
    chunk_size: usize,
) -> PyResult<Bound<'py, PyTuple>> {
    // Release the GIL for the entire computation
    let (vocab, merges) =
        py.detach(|| train_bpe_core(&input_path, vocab_size, &special_tokens, chunk_size));

    // ── Convert vocab: dict[int, bytes] ─────────────────────────────────
    let py_vocab = PyDict::new(py);
    for (id, token_bytes) in &vocab {
        py_vocab.set_item(id, PyBytes::new(py, token_bytes))?;
    }

    // ── Convert merges: list[tuple[bytes, bytes]] ───────────────────────
    let py_merges = PyList::empty(py);
    for (a, b) in &merges {
        let pair = PyTuple::new(py, &[PyBytes::new(py, a), PyBytes::new(py, b)])?;
        py_merges.append(pair)?;
    }

    let result = PyTuple::new(py, &[py_vocab.as_any(), py_merges.as_any()])?;
    Ok(result)
}

#[pyfunction]
fn hello() -> String {
    "Hello from Rust!".to_string()
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    m.add_function(wrap_pyfunction!(run_train_bpe, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_hello() {
        assert_eq!(hello(), "Hello from Rust!");
    }
}
