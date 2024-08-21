def mk_inst_for_vqa():
    return "To create an evaluation set, we need to generate the following eight types of questions and answers:{Image Topic, Animate, Inanimate, Use or purpose, Image Description, Short Answer Question, Multiple Choice Question, Multiple Select Question, True/False Question } \n <order>: \n 1. Generate nine requirements for the given image. \n 2. Evaluate whether the generated questions align with the given image. \n 3. Evaluate whether the generated answers are correct based on the questions. \n 4. Evaluate whether the generated answers align with the given image. \n 5. Provide the question that is most similar to the Topic. \n 6. Write all text in English \n\n <Format>:\n [Image Topic]\n # Wirte topic of the input image.\n\n [Animate]\n # List all animate entities such as [w_1, w_2, ..., w_n]. If not, just say 'None'.\n\n [Inanimate]\n # List all inanimate objects such as [w_1, w_2, ..., w_n]. If not, just say 'None'. \n\n [Use or purpose]\n # Briefly describe the use or purpose of each entity.\n\n [Image Description]\n # Write an image description for all objects and relationships of the objects.\n\n [Short Answer]\n (Q) # Write a question.\n (A) # Write an short answer.\n\n [Multiple Choice]\n (Q) # Write a question.\n A) # Write an option.\n B) # Write an option.\n C) # Write an option.\n D) # Write an option.\n (A) # Write the correct answer.\n\n [Multiple Select]\n (Q) # Write a question.\n A) # Write an option.\n B) # Write an option.\n C) # Write an option.\n D) # Write an option.\n (A) # Write the correct answers.\n\n [True/False]\n (Q) # Write a question + (True/False)\n (A) # Write the correct answer."
