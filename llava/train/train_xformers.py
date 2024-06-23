# Make it more memory efficient by monkey patching the LLaMA model with xformers attention.
import sys, pathlib
new = [pathlib.Path(path).parent.parent.absolute().as_posix() for path in sys.path if pathlib.Path(path).name == 'train' and pathlib.Path(path).parent.name == 'llava']
sys.path.extend(new)

# Need to call this before importing transformers.
from llava.train.llama_xformers_attn_monkey_patch import (
    replace_llama_attn_with_xformers_attn,
)

replace_llama_attn_with_xformers_attn()

from llava.train.train import train

if __name__ == "__main__":
    train()
