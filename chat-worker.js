import {
  pipeline,
  TextStreamer,
  DynamicCache,
  InterruptableStoppingCriteria,
} from "https://esm.sh/@huggingface/transformers@4.1.0";

// Only SmolLM2 is offered here. Qwen3-0.6B-Instruct-ONNX (q4) and
// DeepSeek-R1-Distill-Qwen-1.5B-ONNX (q4f16) were both tested live and
// reliably crash or hang on generation on at least one real browser/GPU
// combination, reproduced even on the reference implementation
// (https://github.com/twinkites/bonsai-garden) running the same models
// on the same machine, so this isn't specific to this port. Revisit if
// transformers.js/onnxruntime-web ships a fix upstream.
// Known-good ChatML template, used only as a fallback if a model's own
// tokenizer_config.json fails to provide one (seen intermittently for
// SmolLM2-135M-Instruct; cause unconfirmed, possibly a caching or
// transient-fetch issue rather than the template genuinely being absent
// upstream). Matches what HuggingFaceTB/SmolLM2-135M-Instruct publishes.
const FALLBACK_CHATML_TEMPLATE =
  "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}" +
  "{{ '<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n' }}{% endif %}" +
  "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}" +
  "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";

const MODELS = {
  "smollm2-135m": {
    id: "HuggingFaceTB/SmolLM2-135M-Instruct",
    dtype: "q4f16",
    label: "SmolLM2 135M",
    params: { max_new_tokens: 100, do_sample: true, temperature: 0.3, top_p: 0.85, repetition_penalty: 1.15 },
  },
};

class PipelineRegistry {
  static cache = new Map();
  static getOrCreate(key, id = null, dtype = null, progress_callback = null) {
    if (!this.cache.has(key)) {
      this.cache.set(key, pipeline("text-generation", id, { device: "webgpu", dtype, progress_callback }));
    }
    return this.cache.get(key);
  }
}

const stopping_criteria = new InterruptableStoppingCriteria();
let past_key_values_cache = null;
let current_model_key = null;
let current_params = null;

function disposePastKeyValues() {
  past_key_values_cache?.dispose?.();
  past_key_values_cache = null;
}

async function check() {
  try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error("WebGPU is not supported in this browser.");
    self.postMessage({ status: "check_ok" });
  } catch (e) {
    self.postMessage({ status: "error", data: e.message });
  }
}

async function listModels() {
  self.postMessage({
    status: "model_list",
    data: Object.entries(MODELS).map(([key, m]) => ({ key, label: m.label })),
  });
}

async function load(modelKey) {
  const entry = MODELS[modelKey];
  if (!entry) {
    self.postMessage({ status: "error", data: `Unknown model: ${modelKey}` });
    return;
  }

  if (current_model_key && current_model_key !== modelKey) disposePastKeyValues();
  current_model_key = modelKey;
  current_params = entry.params;

  self.postMessage({ status: "loading", data: "Fetching model weights..." });

  try {
    const generator = await PipelineRegistry.getOrCreate(modelKey, entry.id, entry.dtype, (info) => {
      if (info.status === "progress_total") {
        self.postMessage({ status: "progress", loaded: Number(info.loaded ?? 0), total: Number(info.total ?? 0) });
      }
    });

    if (!generator.tokenizer.chat_template) {
      generator.tokenizer.chat_template = FALLBACK_CHATML_TEMPLATE;
    }

    self.postMessage({ status: "loading", data: "Warming up model..." });
    const inputs = generator.tokenizer("a");
    await generator.model.generate({ ...inputs, max_new_tokens: 1 });
    self.postMessage({ status: "ready" });
  } catch (e) {
    PipelineRegistry.cache.delete(modelKey);
    self.postMessage({ status: "error", data: e.message });
  }
}

async function generate(messages) {
  const generator = await PipelineRegistry.getOrCreate(current_model_key);

  let startTime;
  let numTokens = 0;
  const streamer = new TextStreamer(generator.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (output) => {
      const tps = numTokens > 1 ? (numTokens / (performance.now() - startTime)) * 1000 : null;
      self.postMessage({ status: "update", output, tps, numTokens });
    },
    token_callback_function: () => {
      startTime ??= performance.now();
      numTokens++;
    },
  });

  self.postMessage({ status: "start" });
  past_key_values_cache ??= new DynamicCache();

  try {
    const output = await generator(messages, {
      ...current_params,
      streamer,
      stopping_criteria,
      past_key_values: past_key_values_cache,
    });
    self.postMessage({ status: "complete", output: output[0].generated_text.at(-1).content });
  } catch (e) {
    self.postMessage({ status: "error", data: e.message });
  }
}

self.addEventListener("message", async ({ data: { type, data } }) => {
  switch (type) {
    case "check":
      check();
      break;
    case "list_models":
      listModels();
      break;
    case "load":
      load(data);
      break;
    case "generate":
      stopping_criteria.reset();
      generate(data.messages);
      break;
    case "interrupt":
      stopping_criteria.interrupt();
      break;
    case "reset":
      disposePastKeyValues();
      stopping_criteria.reset();
      break;
  }
});
