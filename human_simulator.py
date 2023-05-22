
from utils import get_content_between_a_b, parse_instructions,get_api_response

class Human:

    def __init__(self, input, memory, embedder):
        self.input = input
        if memory:
            self.memory = memory
        else:
            self.memory = self.input['output_memory']
        self.embedder = embedder
        self.output = {}


    def prepare_input(self):
        previous_paragraph = self.input["input_paragraph"]
        writer_new_paragraph = self.input["output_paragraph"]
        memory = self.input["output_memory"]
        user_edited_plan = self.input["output_instruction"]

        input_text = f"""
        Now imagine you are a novelist writing a Chinese novel with the help of ChatGPT. You will be given a previously written paragraph (wrote by you), and a paragraph written by your ChatGPT assistant, a summary of the main storyline maintained by your ChatGPT assistant, and a plan of what to write next proposed by your ChatGPT assistant.
    I need you to write:
    1. Extended Paragraph: Extend the new paragraph written by the ChatGPT assistant to twice the length of the paragraph written by your ChatGPT assistant.
    2. Selected Plan: Copy the plan proposed by your ChatGPT assistant.
    3. Revised Plan: Revise the selected plan into an outline of the next paragraph.
    
    Previously written paragraph:  
    {previous_paragraph}

    The summary of the main storyline maintained by your ChatGPT assistant:
    {memory}

    The new paragraph written by your ChatGPT assistant:
    {writer_new_paragraph}

    The plan of what to write next proposed by your ChatGPT assistant:
    {user_edited_plan}

    Now start writing, organize your output by strictly following the output format as below,所有输出仍然保持是中文:
    
    Extended Paragraph: 
    <string of output paragraph>, around 40-50 sentences.

    Selected Plan: 
    <copy the plan here>

    Revised Plan:
    <string of revised plan>, keep it short, around 5-7 sentences.

    Very Important:
    Remember that you are writing a novel. Write like a novelist and do not move too fast when writing the plan for the next paragraph. Think about how the plan can be attractive for common readers when selecting and extending the plan. Remember to follow the length constraints! Remember that the chapter will contain over 10 paragraphs and the novel will contain over 100 chapters. And the next paragraph will be the second paragraph of the second chapter. You need to leave space for future stories.

    """
        return input_text
    
    def parse_plan(self,response):
        plan = get_content_between_a_b('Selected Plan:','Reason',response)
        return plan


    def select_plan(self,response_file):
        
        previous_paragraph = self.input["input_paragraph"]
        writer_new_paragraph = self.input["output_paragraph"]
        memory = self.input["output_memory"]
        previous_plans = self.input["output_instruction"]
        prompt = f"""
    Now imagine you are a helpful assistant that help a novelist with decision making. You will be given a previously written paragraph and a paragraph written by a ChatGPT writing assistant, a summary of the main storyline maintained by the ChatGPT assistant, and 3 different possible plans of what to write next.
    I need you to:
    Select the most interesting and suitable plan proposed by the ChatGPT assistant.

    Previously written paragraph:  
    {previous_paragraph}

    The summary of the main storyline maintained by your ChatGPT assistant:
    {memory}

    The new paragraph written by your ChatGPT assistant:
    {writer_new_paragraph}

    Three plans of what to write next proposed by your ChatGPT assistant:
    {parse_instructions(previous_plans)}

    Now start choosing, organize your output by strictly following the output format as below:
      
    Selected Plan: 
    <copy the selected plan here>

    Reason:
    <Explain why you choose the plan>
    """
        print(prompt+'\n'+'\n')

        response = get_api_response(prompt)

        plan = self.parse_plan(response)
        while plan == None:
            response = get_api_response(prompt)
            plan= self.parse_plan(response)

        if response_file:
            with open(response_file, 'a', encoding='utf-8') as f:
                f.write(f"Selected plan here:\n{response}\n\n")

        return plan
        
    def parse_output(self, text):
        try:
            if text.splitlines()[0].startswith('Extended Paragraph'):
                new_paragraph = get_content_between_a_b(
                    'Extended Paragraph:', 'Selected Plan', text)
            else:
                new_paragraph = text.splitlines()[0]

            lines = text.splitlines()
            if lines[-1] != '\n' and lines[-1].startswith('Revised Plan:'):
                revised_plan = lines[-1][len("Revised Plan:"):]
            elif lines[-1] != '\n':
                revised_plan = lines[-1]

            output = {
                "output_paragraph": new_paragraph,
                # "selected_plan": selected_plan,
                "output_instruction": revised_plan,
                # "memory":self.input["output_memory"]
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
                f.write(f"Human's output here:\n{response}\n\n")
