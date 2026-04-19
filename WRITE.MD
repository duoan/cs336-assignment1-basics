## 2.1
- What Unicode character does chr(0) return?
    - `'\x00'`
- How does this character's string representation (__repr__()) differ from its printed representation?
    - `chr(0).__repr__()` -> `"'\\x00'"`
- What happends when this character occurs in text? It may be helpful to play around with the following in your Python interpreter and see if it matches your expectations:
    ```
    >>> chr(0)
    >>> print(chr(0))
    >>> "this is a test" + chr(0) + "string"
    >>> print("this is a test" + chr(0) + "string")
    ```
    -  `chr(0)` is the null character — a real, non-printable character that lives inside the string but usually renders as nothing in your terminal.

## 2.2
- What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than
UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various
input strings.
    - UTF-8 uses a compact 256-token base vocabulary and encodes ACSII in just 1 byte per character (vs. 2 for UTF-16 and 4 for UTF-32), giving shorter sequences for typical text and avoiding the wasted null-byte padding you'd see in `"hello".encode('utf-32')`. It's also the dominant encoding on the web, so training data is already UTF-8 native and the model generalizes better across languages without bloating sequence length.

- Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.
    ```
    def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
        return "".join([bytes([b]).decode("utf-8") for b in bytestring])

    >>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
    'hello'

    ```
    
    - Example: 
    ```
    >>> "é".encode("utf-8")
    b'\xc3\xa9'
    >>> decode_utf8_bytes_to_str_wrong("é".encode("utf-8"))
    Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
        import platform
    File "<stdin>", line 2, in decode_utf8_bytes_to_str_wrong
        import sys
        
    UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc3 in position 0: unexpected end of data
    ```
    - Why? The function decodes each byte independently, but UTF-8 is a variable-width encoding where a single character can span 1–4 bytes. Any non-ASCII character is encoded as a multi-byte sequence whose individual bytes are not valid standalone UTF-8 — they're continuation bytes (0x80–0xBF) that only have meaning as part of a group. Calling `.decode("utf-8")` on one of those bytes alone raises `UnicodeDecodeError`.



## Problem (adamw_accounting): Resource accouting for training with AdawW

Let us compute how much memory and compute running AdamW requires. Assume we are using
float32 for every tensor.

a). How much peak memory does running AdamW require?

- float32 -> 4 bytes

### Parameters
#### Transformer Block 

> Parameters: 4D + 16D^2 + 4D + 32D^2 => 8D + 48D^2
> Activation: 12*B*S*D + 4*B*S + 20*B*S*D + 8*B*H*S*S + 12*B*S*D + 36*B*S*D
> => 4BS + (12 + 20 + 12 + 36)BSD + 8*B*H*S*S
> => 4BS + 80BSD + 8BHSS

- ln1: RMSNorm: (d_model,) parameters
    - parameters: 4D bytes
    - activation: 12*B*L*D bytes
- attn: MultiHeadSelfAttention
    - parameters: 4*4D*D => 16 D*D bytes
    - activation: 4BS + 20BSD + 8BHSS bytes
    - self.rope: RotaryPositionalEmbedding(theta, self.d_head, max_seq_len) -> buffer: 
    - self.q_proj = Linear(d_model, d_model) -> 
    - self.k_proj = Linear(d_model, d_model) -> 
    - self.v_proj = Linear(d_model, d_model) ->
    - self.output_proj = Linear(d_model, d_model)
- ln2: RMSNorm: (d_model,) parameters
    - parameters: 4D bytes
    - activation: 12*B*S*D bytes
- ffn: 
    - parameters: 8/3 * D * D * 3 * 4 bytes => 32 D*D bytes
    - activation: 36 B*S*D bytes
    - self.w1 = Linear(d_ff, d_model, device=device, dtype=dtype)
    - self.w2 = Linear(d_model, d_ff, device=device, dtype=dtype)
    - self.w3 = Linear(d_ff, d_model, device=device, dtype=dtype)
