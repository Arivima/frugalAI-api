import torch
import logging
import re
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
        quote: str = "Who are you?",
        max_new_tokens: int = 2048,
    ):
        
        # from codecarbon import EmissionsTracker

        # tracker = EmissionsTracker(
        #     measure_power_secs=2,  # we track more frequently than default value (15)
        #     log_level="error"      # we only display error level logs in this notebook
        #     )

        assert self.model is not None

        logger.info(f"LLMWrapper.generate")
        logger.info(f"quote : {quote}")

        self.model.eval()

        import textwrap

        STUDENT_SYSTEM_MSG = textwrap.dedent("""\
            You are a climate statement classifier.
            Your task is to categorize statements by identifying which type of climate narrative they represent.

            ### Categories:
            0 - Not relevant: No climate-related claims or doesn't fit other categories
            1 - Denial: Claims climate change is not happening
            2 - Attribution denial: Claims human activity is not causing climate change
            3 - Impact minimization: Claims climate change impacts are minimal or beneficial
            4 - Solution opposition: Claims solutions to climate change are harmful
            5 - Science skepticism: Challenges climate science validity or methods
            6 - Actor criticism: Attacks credibility of climate scientists or activists
            7 - Fossil fuel promotion: Asserts importance of fossil fuels
            """).strip()

        STUDENT_USER_TEMPLATE = textwrap.dedent("""\
            Classify the following statement into one category (0-7).
            ### Statement to classify:
            {quote}
            ### Answer:
            Category:
            """).strip()

        messages = [
            {"role": "system", "content": STUDENT_SYSTEM_MSG},
            {"role": "user", "content": STUDENT_USER_TEMPLATE.format(quote=quote)},
        ]

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)

        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        answer = output_text.split('assistant')[1]

        logger.info(f"answer: {answer}")

        m = re.search(r"\d", answer)
        if m:
            category = m.group(0)
            explanation = answer.split(category)[1].strip()
        else:
            category = ''
            explanation = answer
        logger.info(f"category: {category}")
        logger.info(f"explanation: {explanation}")

        return category, explanation


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
