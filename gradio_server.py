import gradio as gr

from recurrentgpt import RecurrentGPT
from human_simulator import Human
from revChatGPT.V1 import Chatbot
from sentence_transformers import SentenceTransformer
from utils import get_init, parse_instructions
import re

# from urllib.parse import quote_plus
# from pymongo import MongoClient

# uri = "mongodb://%s:%s@%s" % (quote_plus("xxx"),
#                               quote_plus("xxx"), "localhost")
# client = MongoClient(uri, maxPoolSize=None)
# db = client.recurrentGPT_db
# log = db.log

_CACHE = {}

access_token = 'eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1UaEVOVUpHTkVNMVFURTRNMEZCTWpkQ05UZzVNRFUxUlRVd1FVSkRNRU13UmtGRVFrRXpSZyJ9.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL3Byb2ZpbGUiOnsiZW1haWwiOiJtc2xhYi1jaGF0Z3B0QG91dGxvb2suY29tIiwiZW1haWxfdmVyaWZpZWQiOnRydWV9LCJodHRwczovL2FwaS5vcGVuYWkuY29tL2F1dGgiOnsidXNlcl9pZCI6InVzZXItU1d4aFVmNE5HUFo4b25oakZocW5HczlsIn0sImlzcyI6Imh0dHBzOi8vYXV0aDAub3BlbmFpLmNvbS8iLCJzdWIiOiJ3aW5kb3dzbGl2ZXwwZTAwZGI0YzllZmJjNGYwIiwiYXVkIjpbImh0dHBzOi8vYXBpLm9wZW5haS5jb20vdjEiLCJodHRwczovL29wZW5haS5vcGVuYWkuYXV0aDBhcHAuY29tL3VzZXJpbmZvIl0sImlhdCI6MTY4MjkyMjI5MSwiZXhwIjoxNjg0MTMxODkxLCJhenAiOiJUZEpJY2JlMTZXb1RIdE45NW55eXdoNUU0eU9vNkl0RyIsInNjb3BlIjoib3BlbmlkIHByb2ZpbGUgZW1haWwgbW9kZWwucmVhZCBtb2RlbC5yZXF1ZXN0IG9yZ2FuaXphdGlvbi5yZWFkIG9mZmxpbmVfYWNjZXNzIn0.otOE1JgzTLX12scBH4LWDhIodGwpDNig27RwmqoVqZQcAEzyHuZbNjycQXh6yy8B9QjraGZN31LrB-JBtZRW_dTZkt7ops768A6MVREEQ4FveCQ-K4vSeezq59sOQAAzlau7jxkAMyHFDfAb7fGHgJzGvRN2QfNE-1bZNngvs3Er0N4laHQ1L2n5fehIWZ_I0p95RTK-kNn0ewLsXzmHzbMS2jh_JPt50PXz6My0YLbtJ7l-s6k0eT3vPDo96eBOObtQtzHFOdIsDnOKUZ0IOoMeuKzYfY9Ct_QeqPsxKhHKbScy_-TwSvLNBU_fyJlu9eDFpQjHD66uW1S8ZfpaHw'

# Build the semantic search model
embedder = SentenceTransformer('multi-qa-mpnet-base-cos-v1')
chatbot = Chatbot(config={
    "access_token": access_token
})


def init_prompt(novel_type, description):
    if description == "":
        description = ""
    else:
        description = " about " + description
    return f"""
Please write a {novel_type} novel{description} with 50 chapters. Follow the format below precisely:

Begin with the name of the novel.
Next, write an outline for the first chapter. The outline should describe the background and the beginning of the novel.
Write the first three paragraphs with their indication of the novel based on your outline. Write in a novelistic style and take your time to set the scene.
Write a summary that captures the key information of the three paragraphs.
Finally, write three different instructions for what to write next, each containing around five sentences. Each instruction should present a possible, interesting continuation of the story.
The output format should follow these guidelines:
Name: <name of the novel>
Outline: <outline for the first chapter>
Paragraph 1: <content for paragraph 1>
Paragraph 2: <content for paragraph 2>
Paragraph 3: <content for paragraph 3>
Summary: <content of summary>
Instruction 1: <content for instruction 1>
Instruction 2: <content for instruction 2>
Instruction 3: <content for instruction 3>

Make sure to be precise and follow the output format strictly.

"""

def init(novel_type, description, request: gr.Request):
    if novel_type == "":
        novel_type = "Science Fiction"
    global _CACHE
    cookie = request.headers['cookie']
    cookie = cookie.split('; _gat_gtag')[0]
    # prepare first init
    init_paragraphs = get_init(chatbot, init_prompt(novel_type, description))
    # print(init_paragraphs)
    start_input_to_human = {
        'output_paragraph': init_paragraphs['Paragraph 3'],
        'input_paragraph': '\n\n'.join([init_paragraphs['Paragraph 1'], init_paragraphs['Paragraph 2']]),
        'output_memory': init_paragraphs['Summary'],
        "output_instruction": [init_paragraphs['Instruction 1'], init_paragraphs['Instruction 2'], init_paragraphs['Instruction 3']]
    }

    _CACHE[cookie] = {"start_input_to_human": start_input_to_human,
                      "init_paragraphs": init_paragraphs}
    written_paras = f"""Title: {init_paragraphs['name']}

Outline: {init_paragraphs['Outline']}

Paragraphs:

{start_input_to_human['input_paragraph']}"""
    long_memory = parse_instructions([init_paragraphs['Paragraph 1'], init_paragraphs['Paragraph 2']])
    # short memory, long memory, current written paragraphs, 3 next instructions
    return start_input_to_human['output_memory'], long_memory, written_paras, init_paragraphs['Instruction 1'], init_paragraphs['Instruction 2'], init_paragraphs['Instruction 3']

def step(short_memory, long_memory, instruction1, instruction2, instruction3, current_paras, request: gr.Request, ):
    if current_paras == "":
        return "", "", "", "", "", ""
    global _CACHE
    # print(list(_CACHE.keys()))
    # print(request.headers.get('cookie'))
    cookie = request.headers['cookie']
    cookie = cookie.split('; _gat_gtag')[0]
    cache = _CACHE[cookie]

    if "writer" not in cache:
        start_input_to_human = cache["start_input_to_human"]
        start_input_to_human['output_instruction'] = [
            instruction1, instruction2, instruction3]
        init_paragraphs = cache["init_paragraphs"]
        human = Human(model=chatbot, input=start_input_to_human,
                      memory=None, embedder=embedder)
        human.step()
        start_short_memory = init_paragraphs['Summary']
        writer_start_input = human.output

        # Init writerGPT
        writer = RecurrentGPT(model=chatbot, input=writer_start_input, short_memory=start_short_memory, long_memory=[
            init_paragraphs['Paragraph 1'], init_paragraphs['Paragraph 2']], memory_index=None, embedder=embedder)
        cache["writer"] = writer
        cache["human"] = human
        writer.step()
    else:
        human = cache["human"]
        writer = cache["writer"]
        output = writer.output
        output['output_memory'] = short_memory
        output['output_instruction'] = [
            instruction1, instruction2, instruction3]
        human.input = output
        human.step()
        writer.input = human.output
        writer.step()

    long_memory = [[v] for v in writer.long_memory]
    # short memory, long memory, current written paragraphs, 3 next instructions
    return writer.output['output_memory'], long_memory, current_paras + '\n\n' + writer.output['input_paragraph'], human.output['output_instruction'], *writer.output['output_instruction']


def controled_step(short_memory, long_memory, selected_instruction, current_paras, request: gr.Request, ):
    if current_paras == "":
        return "", "", "", "", "", ""
    global _CACHE
    # print(list(_CACHE.keys()))
    # print(request.headers.get('cookie'))
    cookie = request.headers['cookie']
    cookie = cookie.split('; _gat_gtag')[0]
    cache = _CACHE[cookie]
    if "writer" not in cache:
        start_input_to_human = cache["start_input_to_human"]
        start_input_to_human['output_instruction'] = selected_instruction
        init_paragraphs = cache["init_paragraphs"]
        human = Human(model=chatbot, input=start_input_to_human,
                      memory=None, embedder=embedder)
        human.step_with_edit()
        start_short_memory = init_paragraphs['Summary']
        writer_start_input = human.output

        # Init writerGPT
        writer = RecurrentGPT(model=chatbot, input=writer_start_input, short_memory=start_short_memory, long_memory=[
            init_paragraphs['Paragraph 1'], init_paragraphs['Paragraph 2']], memory_index=None, embedder=embedder)
        cache["writer"] = writer
        cache["human"] = human
        writer.step()
    else:
        human = cache["human"]
        writer = cache["writer"]
        output = writer.output
        output['output_memory'] = short_memory
        output['output_instruction'] = selected_instruction
        human.input = output
        human.step_with_edit()
        writer.input = human.output
        writer.step()

    # short memory, long memory, current written paragraphs, 3 next instructions
    return writer.output['output_memory'], parse_instructions(writer.long_memory), current_paras + '\n\n' + writer.output['input_paragraph'], *writer.output['output_instruction']


# SelectData is a subclass of EventData
def on_select(instruction1, instruction2, instruction3, evt: gr.SelectData):
    selected_plan = int(evt.value.replace("Instruction ", ""))
    selected_plan = [instruction1, instruction2, instruction3][selected_plan-1]
    return selected_plan


with gr.Blocks(title="RecurrentGPT", css="footer {visibility: hidden}", theme="default") as demo:
    gr.Markdown(
        """
    # RecurrentGPT
    Interactive Generation of (Arbitrarily) Long Texts with Human-in-the-Loop
    """)
    with gr.Tab("Auto-Generation"):
        with gr.Row():
            with gr.Column():
                with gr.Box():
                    with gr.Row():
                        with gr.Column(scale=1, min_width=200):
                            novel_type = gr.Textbox(
                                label="Novel Type", placeholder="e.g. science fiction")
                        with gr.Column(scale=2, min_width=400):
                            description = gr.Textbox(label="Description")
                btn_init = gr.Button(
                    "Init Novel Generation", variant="primary")
                gr.Examples(["Science Fiction", "Romance", "Mystery", "Fantasy",
                            "Historical", "Horror", "Thriller", "Western", "Young Adult", ], inputs=[novel_type])
                written_paras = gr.Textbox(
                    label="Written Paragraphs (editable)", max_lines=21, lines=21)
            with gr.Column():
                with gr.Box():
                    gr.Markdown("### Memory Module\n")
                    short_memory = gr.Textbox(
                        label="Short-Term Memory (editable)", max_lines=3, lines=3)
                    long_memory = gr.Textbox(
                        label="Long-Term Memory (editable)", max_lines=6, lines=6)
                    # long_memory = gr.Dataframe(
                    #     # label="Long-Term Memory (editable)",
                    #     headers=["Long-Term Memory (editable)"],
                    #     datatype=["str"],
                    #     row_count=3,
                    #     max_rows=3,
                    #     col_count=(1, "fixed"),
                    #     type="array",
                    # )
                with gr.Box():
                    gr.Markdown("### Instruction Module\n")
                    with gr.Row():
                        instruction1 = gr.Textbox(
                            label="Instruction 1 (editable)", max_lines=4, lines=4)
                        instruction2 = gr.Textbox(
                            label="Instruction 2 (editable)", max_lines=4, lines=4)
                        instruction3 = gr.Textbox(
                            label="Instruction 3 (editable)", max_lines=4, lines=4)
                    selected_plan = gr.Textbox(
                        label="Revised Instruction (from last step)", max_lines=2, lines=2)

                btn_step = gr.Button("Next Step", variant="primary")

        btn_init.click(init, inputs=[novel_type, description], outputs=[
            short_memory, long_memory, written_paras, instruction1, instruction2, instruction3])
        btn_step.click(step, inputs=[short_memory, long_memory, instruction1, instruction2, instruction3, written_paras], outputs=[
            short_memory, long_memory, written_paras, selected_plan, instruction1, instruction2, instruction3])

    with gr.Tab("Human-in-the-Loop"):
        with gr.Row():
            with gr.Column():
                with gr.Box():
                    with gr.Row():
                        with gr.Column(scale=1, min_width=200):
                            novel_type = gr.Textbox(
                                label="Novel Type", placeholder="e.g. science fiction")
                        with gr.Column(scale=2, min_width=400):
                            description = gr.Textbox(label="Description")
                btn_init = gr.Button(
                    "Init Novel Generation", variant="primary")
                gr.Examples(["Science Fiction", "Romance", "Mystery", "Fantasy",
                            "Historical", "Horror", "Thriller", "Western", "Young Adult", ], inputs=[novel_type])
                written_paras = gr.Textbox(
                    label="Written Paragraphs (editable)", max_lines=23, lines=23)
            with gr.Column():
                with gr.Box():
                    gr.Markdown("### Memory Module\n")
                    short_memory = gr.Textbox(
                        label="Short-Term Memory (editable)", max_lines=3, lines=3)
                    long_memory = gr.Textbox(
                        label="Long-Term Memory (editable)", max_lines=6, lines=6)
                with gr.Box():
                    gr.Markdown("### Instruction Module\n")
                    with gr.Row():
                        instruction1 = gr.Textbox(
                            label="Instruction 1", max_lines=3, lines=3, interactive=False)
                        instruction2 = gr.Textbox(
                            label="Instruction 2", max_lines=3, lines=3, interactive=False)
                        instruction3 = gr.Textbox(
                            label="Instruction 3", max_lines=3, lines=3, interactive=False)
                    with gr.Row():
                        with gr.Column(scale=1, min_width=100):
                            selected_plan = gr.Radio(["Instruction 1", "Instruction 2", "Instruction 3"], label="Instruction Selection",)
                                                    #  info="Select the instruction you want to revise and use for the next step generation.")
                        with gr.Column(scale=3, min_width=300):
                            selected_instruction = gr.Textbox(
                                label="Selected Instruction (editable)", max_lines=5, lines=5)

                btn_step = gr.Button("Next Step", variant="primary")

        btn_init.click(init, inputs=[novel_type, description], outputs=[
            short_memory, long_memory, written_paras, instruction1, instruction2, instruction3])
        btn_step.click(controled_step, inputs=[short_memory, long_memory, selected_instruction, written_paras], outputs=[
            short_memory, long_memory, written_paras, instruction1, instruction2, instruction3])
        selected_plan.select(on_select, inputs=[
                             instruction1, instruction2, instruction3], outputs=[selected_instruction])

    demo.queue(concurrency_count=1)

if __name__ == "__main__":
    demo.launch(server_port=8003, share=True,
                server_name="0.0.0.0", show_api=False)
