from api.utils import ask_question, split_answer
from rest_framework.decorators import api_view
from rest_framework.response import Response

@api_view(['GET'])
def ask(request):
    question = request.query_params.get('question', 'Who is Adam McKerlie?')
    answer = ask_question(question)

    result = split_answer(answer)
    return Response({'question': question, 'answer': result})