#!/bin/bash
export OPENAI_API_KEY="your_api_key"
iteration=10
outfile=response.txt
init_prompt=init_prompt.json
topic=Aliens
type="science-fiction"
max_tokens=4096

options="\
        --iter $iteration\
        --r_file $outfile \
        --init_prompt $init_prompt \
        --topic $topic \
        --type $type \
        --max-tokens $max_tokens \
        "
python main.py $options