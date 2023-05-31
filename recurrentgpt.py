from utils import get_content_between_a_b, get_api_response
import torch

import random

from sentence_transformers import  util


class RecurrentGPT:

    def __init__(self, input, short_memory, long_memory, memory_index, embedder):
        self.input = input
        self.short_memory = short_memory
        self.long_memory = long_memory
        self.embedder = embedder
        if self.long_memory and not memory_index:
            self.memory_index = self.embedder.encode(
                self.long_memory, convert_to_tensor=True)
        self.output = {}

    def prepare_input(self, new_character_prob=0.1, top_k=2):

        input_paragraph = self.input["output_paragraph"]
        input_instruction = self.input["output_instruction"]

        instruction_embedding = self.embedder.encode(
            input_instruction, convert_to_tensor=True)

        # get the top 3 most similar paragraphs from memory

        memory_scores = util.cos_sim(
            instruction_embedding, self.memory_index)[0]
        top_k_idx = torch.topk(memory_scores, k=top_k)[1]
        top_k_memory = [self.long_memory[idx] for idx in top_k_idx]
        # combine the top 3 paragraphs
        input_long_term_memory = '\n'.join(
            [f"Related Paragraphs {i+1} :" + selected_memory for i, selected_memory in enumerate(top_k_memory)])
        # randomly decide if a new character should be introduced
        if random.random() < new_character_prob:
            new_character_prompt = f"If it is reasonable, you can introduce a new character in the output paragrah and add it into the memory."
        else:
            new_character_prompt = ""

        input_text = f"""I need you to help me write a novel. Now I give you a memory (a brief summary) of 400 words, you should use it to store the key content of what has been written so that you can keep track of very long context. For each time, I will give you your current memory (a brief summary of previous stories. You should use it to store the key content of what has been written so that you can keep track of very long context), the previously written paragraph, and instructions on what to write in the next paragraph. 
    I need you to write:
    1. Output Paragraph: the next paragraph of the novel. The output paragraph should contain around 20 sentences and should follow the input instructions.
    2. Output Memory: The updated memory. You should first explain which sentences in the input memory are no longer necessary and why, and then explain what needs to be added into the memory and why. After that you should write the updated memory. The updated memory should be similar to the input memory except the parts you previously thought that should be deleted or added. The updated memory should only store key information. The updated memory should never exceed 20 sentences!
    3. Output Instruction:  instructions of what to write next (after what you have written). You should output 3 different instructions, each is a possible interesting continuation of the story. Each output instruction should contain around 5 sentences
    Here are the inputs: 

    Input Memory:  
    {self.short_memory}

    Input Paragraph:
    {input_paragraph}

    Input Instruction:
    {input_instruction}

    Input Related Paragraphs:
    {input_long_term_memory}
    
    Now start writing, organize your output by strictly following the output format as below:
    Output Paragraph: 
    <string of output paragraph>, around 20 sentences.

    Output Memory: 
    Rational: <string that explain how to update the memory>;
    Updated Memory: <string of updated memory>, around 10 to 20 sentences

    Output Instruction: 
    Instruction 1: <content for instruction 1>, around 5 sentences
    Instruction 2: <content for instruction 2>, around 5 sentences
    Instruction 3: <content for instruction 3>, around 5 sentences

    Very important!! The updated memory should only store key information. The updated memory should never contain over 500 words!
    Finally, remember that you are writing a novel. Write like a novelist and do not move too fast when writing the output instructions for the next paragraph. Remember that the chapter will contain over 10 paragraphs and the novel will contain over 100 chapters. And this is just the beginning. Just write some interesting staffs that will happen next. Also, think about what plot can be attractive for common readers when writing output instructions. 

    Very Important: 
    You should first explain which sentences in the input memory are no longer necessary and why, and then explain what needs to be added into the memory and why. After that, you start rewrite the input memory to get the updated memory. 
    {new_character_prompt}
    """
        return input_text

    def parse_output(self, output):
        try:
            output_paragraph = get_content_between_a_b(
                'Output Paragraph:', 'Output Memory', output)
            output_memory_updated = get_content_between_a_b(
                'Updated Memory:', 'Output Instruction:', output)
            self.short_memory = output_memory_updated
            ins_1 = get_content_between_a_b(
                'Instruction 1:', 'Instruction 2', output)
            ins_2 = get_content_between_a_b(
                'Instruction 2:', 'Instruction 3', output)
            lines = output.splitlines()
            # content of Instruction 3 may be in the same line with I3 or in the next line
            if lines[-1] != '\n' and lines[-1].startswith('Instruction 3'):
                ins_3 = lines[-1][len("Instruction 3:"):]
            elif lines[-1] != '\n':
                ins_3 = lines[-1]

            output_instructions = [ins_1, ins_2, ins_3]
            assert len(output_instructions) == 3

            output = {
                "input_paragraph": self.input["output_paragraph"],
                "output_memory": output_memory_updated,  # feed to human
                "output_paragraph": output_paragraph,
                "output_instruction": [instruction.strip() for instruction in output_instructions]
            }

            return output
        except:
            return None

    def step(self, response_file=None):

        prompt = self.prepare_input()

        print(prompt+'\n'+'\n')

        response = get_api_response(prompt)

        self.output = self.parse_output(response)
        while self.output == None:
            response = get_api_response(prompt)
            self.output = self.parse_output(response)
        if response_file:
            with open(response_file, 'a', encoding='utf-8') as f:
                f.write(f"Writer's output here:\n{response}\n\n")

        self.long_memory.append(self.input["output_paragraph"])
        self.memory_index = self.embedder.encode(
            self.long_memory, convert_to_tensor=True)
