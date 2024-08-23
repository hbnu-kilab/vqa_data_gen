def mk_inst_for_vqa():
    return "To create an evaluation set, we need to generate the following eight types of questions and answers:{Image Topic, Animate, Inanimate, Use or purpose, Image Description, Short Answer Question, Multiple Choice Question, Multiple Select Question, True/False Question } \n <order>: \n 1. Generate nine requirements for the given image. \n 2. Evaluate whether the generated questions align with the given image. \n 3. Evaluate whether the generated answers are correct based on the questions. \n 4. Evaluate whether the generated answers align with the given image. \n 5. Provide the question that is most similar to the Topic. \n 6. Write all text in English \n\n <Format>:\n [Image Topic]\n # Wirte topic of the input image.\n\n [Animate]\n # List all animate entities such as [w_1, w_2, ..., w_n]. If not, just say 'None'.\n\n [Inanimate]\n # List all inanimate objects such as [w_1, w_2, ..., w_n]. If not, just say 'None'. \n\n [Use or purpose]\n # Briefly describe the use or purpose of each entity.\n\n [Image Description]\n # Write an image description for all objects and relationships of the objects.\n\n [Short Answer Question]\n (Q) # Write a question.\n (A) # Write an short answer.\n\n [Multiple Choice Question]\n (Q) # Write a question.\n A) # Write an option.\n B) # Write an option.\n C) # Write an option.\n D) # Write an option.\n (A) # Write the correct answer.\n\n [Multiple Select Question]\n (Q) # Write a question.\n A) # Write an option.\n B) # Write an option.\n C) # Write an option.\n D) # Write an option.\n (A) # Write the correct answers.\n\n [True/False Question]\n (Q) # Write a question + (True/False)\n (A) # Write the correct answer."

def mk_inst_for_vqa_ko():
    return ' '.join(f"평가 세트를 생성하려면 다음의 9가지 유형의 질문과 답변을 생성해야 합니다: (Image Topic, Animate, Inanimate, Use or purpose, Image Description, Short Answer Question, Multiple Choice Question, Multiple Select Question, True/False Question)\n\n\
    <order>:\n\
    1. 주어진 이미지에 대한 9가지 요구 사항을 생성합니다.\n\
    2. 생성된 질문이 주어진 이미지와 일치하는지 평가합니다.\n\
    3. 생성된 답변이 질문에 따라 올바른지 평가합니다.\n\
    4. 생성된 답변이 주어진 이미지와 일치하는지 평가합니다.\n\
    5. 주제와 가장 유사한 질문을 제공합니다.\n\
    6. 모든 텍스트를 한국어로 작성합니다.\n\n\
    <Format>:\n\
    [Image Topic]\n\
    # 입력 이미지의 주제를 작성하세요.\n\n\
    [Animate]\n\
    # [w_1, w_2, ..., w_n]과 같은 모든 animate entity를 나열하세요. 그렇지 않으면 'None'이라고만 말하세요.\n\n\
    [Inanimate]\n\
    # [w_1, w_2, ..., w_n]과 같은 모든 inanimate entity를 나열하세요. 그렇지 않으면 'None'이라고만 말하세요.\n\n\
    [Use or purpose]\n\
    # 각 entity의 용도나 목적을 간략하게 설명하세요.\n\n\
    [Image Description]\n\
    # 모든 객체와 객체 간 관계에 대한 이미지 설명을 작성하세요.\n\n\
    [Short Answer]\n\
    (Q) # 단답형 질문을 작성하세요.\n\
    (A) # 단답형 정답을 작성하세요.\n\n\
    [Multiple Choice]\n\
    (Q) # 질문을 작성하세요.\n\
    A) # 보기를 작성하세요.\n\
    B) # 보기를 작성하세요.\n\
    C) # 보기를 작성하세요.\n\
    D) # 보기를 작성하세요.\n\
    (A) # 보기 중 올바른 정답을 작성하세요.\n\n\
    [Multiple Select]\n\
    (Q) # 질문을 작성하세요.\n\
    A) # 보기를 작성하세요.\n\
    B) # 보기를 작성하세요.\n\
    C) # 보기를 작성하세요.\n\
    D) # 보기를 작성하세요.\n\
    (A) # 보기 중 올바른 정답을 모두 작성하세요.\n\n\
    [True/False]\n\
    (Q) # 질문을 작성하세요 + (True/False)\n\
    (A) # 정답을 작성하세요.".split())


def mk_vqa_for_multiple_choice(question, choice):
    return ' '.join(f"Generate appropriate answers by looking at the given image and questions. There are three question types: (Animate, Inanimate, Multiple Choice)\n\
    [Animate] \
    # List all animate entities such as [w_1, w_2, ..., w_n]. If not, just say 'None'.\n\n\
    [Inanimate] \
    # List all inanimate objects such as [w_1, w_2, ..., w_n]. If not, just say 'None'.\n\n\
    [Multiple Choice]\n\
    (Q) {question}\n\
    {choice[0]}\n\
    {choice[1]}\n\
    {choice[2]}\n\
    {choice[3]}\n\
    (A) # Look at the question and answer above and choose the correct answer.".split())