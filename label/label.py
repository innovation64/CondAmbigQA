import gradio as gr
import json
import os
import portalocker

# Read the JSON file from the specified path
def load_json():
    file_path = "filter.json"  # Replace with the actual file path
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def update_message(request: gr.Request):
    return f"Welcome, {request.username}"

# Read the saved annotation file
def load_annotations():
    save_path = "filter_annotations.json"  # Replace with the actual save path
    annotations = []
    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as f:
            portalocker.lock(f, portalocker.LOCK_SH)  # Acquire a shared lock
            try:
                annotations = json.load(f)
            except json.JSONDecodeError:
                annotations = []
            portalocker.unlock(f)  # Release the lock
    return annotations

def display_question(data, idx):
    question = data[idx].get("question", "")
    return question

def display_contexts(data, idx, start=0):
    contexts = data[idx].get("ctxs", [])
    end = min(start + 5, len(contexts))  # Display 5 contexts at a time
    return contexts[start:end]

def update_contexts(start, increment, total):
    start += increment
    if start < 0:
        start = 0
    elif start >= total:
        start = max(0, total - 5)  # Ensure 'start' does not exceed the range and at least 5 contexts are displayed
    return start

def update_display(data, idx, start):
    total = len(data[idx].get("ctxs", []))
    end = min(start + 5, total)  # Ensure not to exceed the range
    ctxs = data[idx].get("ctxs", [])[start:end]
    ctxs_text = [f"{i+1}. Title: {ctx['title']}\nText: {ctx['text']}" for i, ctx in enumerate(ctxs, start=start)]
    return ctxs_text

def save_annotations(annotations):
    save_path = "filter_annotations.json"  # Replace with the actual save path
    with open(save_path, 'w', encoding='utf-8') as f:
        portalocker.lock(f, portalocker.LOCK_EX)  # Acquire an exclusive lock
        json.dump(annotations, f, ensure_ascii=False, indent=4)
        portalocker.unlock(f)  # Release the lock
    return annotations

# Load the data and annotations
data = load_json()
annotations = load_annotations()

with gr.Blocks() as demo:
    m = gr.Markdown()
    demo.load(update_message, None, m)
    gr.Markdown("## RAG Logic Context Annotator")
    
    with gr.Row():
        with gr.Column():
            idx_slider = gr.Slider(0, len(data) - 1, step=1, label="Select Question Index")
            question_output = gr.Textbox(label="Question", interactive=False)
            ctxs_output = [gr.Textbox(label=f"Context {i+1}", interactive=False) for i in range(5)]  # Create 5 Textbox components
            with gr.Row():
                prev_button = gr.Button("Previous 5")
                next_button = gr.Button("Next 5")
        
        with gr.Column():
            mark_contexts = gr.Textbox(label="Marked Contexts", interactive=True)
            concept1 = gr.Textbox(label="Concept 1")
            concept2 = gr.TextArea(label="Concept 2 (enter each answer on a new line)")
            references = gr.Dropdown(label="References", choices=[], multiselect=True)
            reason = gr.Textbox(label="Reason for Annotation")
            properties = gr.State([])  # Used to store multiple properties
            add_property_button = gr.Button("Add Property")
            delete_property_dropdown = gr.Dropdown(label="Select Property to Delete", choices=[])
            delete_property_button = gr.Button("Delete Property")
            properties_output = gr.DataFrame(label="Properties", headers=["Concept", "Concept2", "Citation"], interactive=False)
            answer = gr.Textbox(label="Answer")
            ambiguity_dropdown = gr.Dropdown(label="Ambiguity Options", choices=["No Ambiguity", "Cannot Annotate", "No Answer in Context", "Ambiguous Question"])
            save_button = gr.Button("Save Annotations")
            mark_complete_button = gr.Button("Mark Complete and Next")
    
    start_state = gr.State(0)

    def update_ui(idx, annotations):
        start = 0  # Reset to 0
        annotations = load_annotations()  # Reload annotations each time the UI is updated
        question = display_question(data, idx)
        ctxs_text = update_display(data, idx, start)
        existing_annotation = next((ann for ann in annotations if ann["Ambiguous Question"] == question), None)
        if existing_annotation:
            properties = existing_annotation["Property"]
            answer = existing_annotation["Answer"]
        else:
            properties = []
            answer = ""
        
        property_choices = []
        for prop in properties:
            concept = prop['Concept']
            concept2 = ','.join(prop['Concept2'])
            citations_list = [f"{c['Title']}:{c['Text']}" for c in prop['citation']]
            citations = ', '.join(citations_list)
            reason = prop.get('Reason', '')
            property_choices.append(f"{concept} - {concept2} - {citations} - {reason}")
        
        properties_table = []
        for prop in properties:
            concept = prop['Concept']
            concept2 = ','.join(prop['Concept2'])
            citations_list = [f"{c['Title']}:{c['Text']}" for c in prop['citation']]
            citations = ', '.join(citations_list)
            reason = prop.get('Reason', '')
            properties_table.append([concept, concept2, citations, reason])
        
        citations = [f"{i+1}. {ctx['title']} - {ctx['text']}" for i, ctx in enumerate(data[idx].get("ctxs", []))]
        
        return [question] + ctxs_text + [properties, answer, gr.update(choices=property_choices), properties_table, gr.update(choices=citations), "", "", "", properties, start]

    idx_slider.change(
        update_ui,
        inputs=[idx_slider, gr.State(annotations)],
        outputs=[question_output] + ctxs_output + [properties, answer, delete_property_dropdown, properties_output, references, mark_contexts, concept1, concept2, properties, start_state]
    )

    prev_button.click(
        lambda start: update_contexts(start, -5, len(data[idx_slider.value].get("ctxs", []))),
        inputs=start_state,
        outputs=start_state
    )
    next_button.click(
        lambda start: update_contexts(start, 5, len(data[idx_slider.value].get("ctxs", []))),
        inputs=start_state,
        outputs=start_state
    )

    start_state.change(
        lambda idx, start: update_display(data, idx, start),
        inputs=[idx_slider, start_state],
        outputs=ctxs_output
    )
    
    def add_property_handler(concept1, concept2, references, properties, reason):
        concept2_list = concept2.split("\n")  # Split the multiline text into a list
        new_property = {
            "Concept": concept1,
            "Concept2": concept2_list,
            "citation": [
                {"Title": ref.split(" - ")[0], "Text": " - ".join(ref.split(" - ")[1:])}
                for ref in references if " - " in ref
            ],
            "Reason": reason
        }
        # Check if the property already exists and update it
        for prop in properties:
            if prop["Concept"] == concept1:
                prop["Concept2"] = concept2_list
                prop["citation"] = [
                    {"Title": ref.split(" - ")[0], "Text": " - ".join(ref.split(" - ")[1:])}
                    for ref in references if " - " in ref
                ]
                prop["Reason"] = reason
                break
        else:
            properties.append(new_property)
        
        property_choices = []
        for prop in properties:
            concept = prop['Concept']
            concept2 = ','.join(prop['Concept2'])
            citation_list = [f"{c['Title']}:{c['Text']}" for c in prop['citation']]
            citations = ', '.join(citation_list)
            reason = prop.get('Reason', '')
            property_choices.append(f"{concept} - {concept2} - {citations} - {reason}")
        
        properties_table = []
        for prop in properties:
            concept = prop['Concept']
            concept2 = ','.join(prop['Concept2'])
            citation_list = [f"{c['Title']}:{c['Text']}" for c in prop['citation']]
            citations = ', '.join(citation_list)
            reason = prop.get('Reason', '')
            properties_table.append([concept, concept2, citations, reason])
        
        return properties, gr.update(choices=property_choices), properties_table, gr.update(value=""), gr.update(value=""), gr.update(value="")

    add_property_button.click(
        add_property_handler,
        inputs=[concept1, concept2, references, properties, reason],
        outputs=[properties, delete_property_dropdown, properties_output, references, concept1, concept2]
    )

    def delete_property_handler(properties, delete_property_dropdown, idx_slider, annotations):
        if delete_property_dropdown:
            selected_property = delete_property_dropdown.split(" - ")
            new_properties = []
            for prop in properties:
                concept2_str = ",".join(prop["Concept2"])
                citations_list = [f"{c['Title']}:{c['Text']}" for c in prop['citation']]
                citations_str = ", ".join(citations_list)
                reason = prop.get('Reason', '')
                if not (prop["Concept"] == selected_property[0] and
                        concept2_str == selected_property[1] and
                        citations_str == selected_property[2] and
                        reason == selected_property[3]):
                    new_properties.append(prop)
            properties = new_properties

        property_choices = []
        for prop in properties:
            concept = prop['Concept']
            concept2 = ','.join(prop['Concept2'])
            citations_list = [f"{c['Title']}:{c['Text']}" for c in prop['citation']]
            citations = ', '.join(citations_list)
            reason = prop.get('Reason', '')
            property_choices.append(f"{concept} - {concept2} - {citations} - {reason}")

        properties_table = []
        for prop in properties:
            concept = prop['Concept']
            concept2 = ','.join(prop['Concept2'])
            citations_list = [f"{c['Title']}:{c['Text']}" for c in prop['citation']]
            citations = ', '.join(citations_list)
            reason = prop.get('Reason', '')
            properties_table.append([concept, concept2, citations, reason])

        # Save the updated annotations after deletion
        question = display_question(data, idx_slider)  # Use idx_slider as the index directly
        existing_annotation = next((ann for ann in annotations if ann["Ambiguous Question"] == question), None)
        if existing_annotation:
            existing_annotation["Property"] = properties
        else:
            annotations.append({
                "Ambiguous Question": question,
                "Property": properties,
                "Answer": ""
            })
        annotations = save_annotations(annotations)

        return properties, gr.update(choices=property_choices), properties_table, gr.update(value=""), gr.update(choices=property_choices)

    delete_property_button.click(
        delete_property_handler,
        inputs=[properties, delete_property_dropdown, idx_slider, gr.State(annotations)],
        outputs=[properties, delete_property_dropdown, properties_output, delete_property_dropdown, delete_property_dropdown]
    )
    
    def update_properties(properties):
        properties_table = []
        for prop in properties:
            concept = prop['Concept']
            concept2 = ','.join(prop['Concept2'])
            citations_list = [f"{c['Title']}:{c['Text']}" for c in prop['citation']]
            citations = ', '.join(citations_list)
            reason = prop.get('Reason', '')
            properties_table.append([concept, concept2, citations, reason])
        return properties_table

    properties.change(update_properties, inputs=properties, outputs=properties_output)
    
    def save_annotation_handler(question_output, properties, answer, annotations, ambiguity_dropdown):
        question = question_output
        # Check if an annotation for this question already exists, and if so, overwrite it
        existing_annotation = next((ann for ann in annotations if ann["Ambiguous Question"] == question), None)
        if existing_annotation:
            existing_annotation["Property"] = properties
            existing_annotation["Answer"] = answer
            existing_annotation["Ambiguity"] = ambiguity_dropdown
        else:
            new_annotation = {
                "Ambiguous Question": question,
                "Property": properties,
                "Answer": answer,
                "Ambiguity": ambiguity_dropdown
            }
            annotations.append(new_annotation)
        annotations = save_annotations(annotations)  # Save the annotations
        return annotations

    save_button.click(
        save_annotation_handler,
        inputs=[question_output, properties, answer, gr.State(annotations), ambiguity_dropdown],
        outputs=[gr.State(annotations)]
    )
    
    def mark_complete_and_next_handler(idx, properties, answer, annotations, ambiguity_dropdown):
        question = data[idx]["question"]
        annotations = save_annotation_handler(question, properties, answer, annotations, ambiguity_dropdown)
        next_idx = idx + 1 if idx < len(data) - 1 else idx
        start = 0  # Reset to 0
        return next_idx, annotations, [], [], gr.update(value=""), gr.update(value=""), gr.update(value=""), [], start

    mark_complete_button.click(
        mark_complete_and_next_handler,
        inputs=[idx_slider, properties, answer, gr.State(annotations), ambiguity_dropdown],
        outputs=[idx_slider, gr.State(annotations), properties, references, mark_contexts, concept1, concept2, properties, start_state]
    )

demo.launch(auth=[("admin", "admin")])
