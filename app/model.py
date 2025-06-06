import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from app.config import setup_logging, Config

logger = logging.getLogger(__name__)


class LLMWrapper:

    def __init__(self):
        try:
            logger.info(f"LLMWrapper.__init__ : Loading model and tokenizer")
            adapter_dir = Config.DESTINATION_DIRECTORY + '/' + Config.ADAPTER_NAME
            self.model_name = Config.MODEL_NAME
            logger.info(f"Base model: {self.model_name}")
            logger.info(f"Adapter directory: {adapter_dir}")

            if torch.cuda.is_available():
                self.device = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
            torch_dtype = torch.float16 if self.device=="mps" else torch.float32
            device_map="auto" if self.device!="cpu" else None
            logger.info(f"Using device={self.device}, dtype={torch_dtype}, device_map={device_map}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map=device_map
            ).to(self.device)

            self.model = PeftModel.from_pretrained(
                self.base_model,
                adapter_dir,
                torch_dtype=torch_dtype,
                device_map=device_map,
            ).to(self.device)

            logger.info(f"✅ model loaded on {next(self.model.parameters()).device}")

        except Exception as e:
            logger.exception(f"❌ Error loading model: {e}.")


    def generate(
        self,
        prompt: str = "Who are you?",
        max_new_tokens: int = 128,
    ):
        assert self.model is not None

        logger.info(f"LLMWrapper.generate")
        logger.info(f"prompt : {prompt}")

        self.model.eval()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        logger.info(f"output_text {output_text}")

        return output_text


    def clear(self):
        try:
            del self.model
            del self.tokenizer

            if self.device == "cuda":
                torch.cuda.empty_cache()
            elif self.device == "mps":
                torch.mps.empty_cache()

        except Exception:
            pass


if __name__ == "__main__":
    setup_logging()
    llm = LLMWrapper()
    print(llm.generate())
