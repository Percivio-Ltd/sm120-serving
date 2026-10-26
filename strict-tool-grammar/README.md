# Strict-tool grammar fix for Qwen tool calls without arguments in vLLM

With strict tool calling (`"strict": true` on a tool, or `--tool-strict-level parameter` on the server), vLLM's `qwen3_coder` and `qwen3_xml` tool parsers constrain each call with xgrammar's built-in Qwen structural tag. That tag frames every call as `<tool_call>\n<function=NAME>\n`, the arguments, then `\n</function>\n</tool_call>`. Qwen's chat template renders a call without arguments as `<function=NAME>\n</function>`, so for a tool such as `run_tests()` the template's own close is illegal and only whitespace can follow. Decoding is then forced through whitespace until a newline happens to win, or until the length limit. The whitespace is inside the tool call, so the client sees no output while it runs.

The bug is reported upstream as [vllm-project/vllm#50989](https://github.com/vllm-project/vllm/issues/50989) and [mlc-ai/xgrammar#802](https://github.com/mlc-ai/xgrammar/issues/802), and [mlc-ai/xgrammar#1004](https://github.com/mlc-ai/xgrammar/pull/1004) proposes the equivalent fix in xgrammar. All three were open on 27 October 2026, and vLLM main still pins `xgrammar == 0.2.8`.

## The fix

[`structural_tag_registry.py`](structural_tag_registry.py) post-processes the structural tag that xgrammar builds for `qwen_3_coder` and `qwen_3_5`. Each Qwen XML call whose argument schema accepts `{}` may close as the template renders it, `</function>\n</tool_call>`. If the schema also accepts arguments, the call's content becomes either the arguments followed by the newline the original close began with, or no arguments. Calls of tools that require arguments, and schemas with combinators (`anyOf`, `oneOf`, `allOf`, `not`, `if`, `$ref`, `dependentRequired`), keep the original grammar exactly. Strictness stays on, so the model cannot invent arguments for a tool that takes none.

Measured with vLLM nightly `0.32.1rc1.dev435+gc50aca144`, xgrammar 0.2.8 and strict tools, on Qwen3.8-27B and Qwen3.8-Flash:

- 3,231 calls with arguments from our agent logs are accepted by both grammars, with identical token masks at all 346,769 positions.
- 430 template-rendered calls without arguments from the same logs: 0 accepted before, 430 after.
- A frozen agent turn that ran to its 4,096-token cap in hidden whitespace closes after 151 tokens with arguments `{}` (3 of 3; the earlier token ids are identical).
- In an agent battery on Qwen3.8-Flash NVFP4 (two RTX PRO 6000, tensor parallel), hidden runaways after a `run_tests {}` call went from 2 of 36 episodes (44K and 57K hidden tokens, each ending at the 262,144-token model length) to 0 of 18. The 25 calls without arguments in the patched episodes closed after a median of 17 tokens outside the reasoning (max 42).

## Files

| File | |
|---|---|
| [`structural_tag_registry.py`](structural_tag_registry.py) | Drop-in replacement for `vllm/tool_parsers/structural_tag_registry.py` |
| [`structural_tag_registry.diff`](structural_tag_registry.diff) | The same change as a diff against vLLM commit [`c50aca1`](https://github.com/vllm-project/vllm/commit/c50aca14416ef29f6ed0ed256302f30646726dcd); `git apply structural_tag_registry.diff` in a vLLM checkout |

## Use

Mount the file over the module in the vLLM image and serve with a Qwen XML tool parser:

```bash
docker run ... \
  -v "$PWD/strict-tool-grammar/structural_tag_registry.py:/usr/local/lib/python3.12/dist-packages/vllm/tool_parsers/structural_tag_registry.py:ro" \
  vllm/vllm-openai@sha256:3379c8e773c619972cc81e9f1db648c38ab5029369d71c7a24149753666c6480 \
  ... --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

`AGENT=1` [`../qwen3.8-27b-nvfp4-all/serve.sh`](../qwen3.8-27b-nvfp4-all/serve.sh) does this. The patch is tested only on the image above. The module is unchanged on vLLM main as of 27 October 2026 (last changed in `83773ecd`), so the diff also applies there. It is no longer needed once vLLM ships an xgrammar release with #1004 or an equivalent fix.
