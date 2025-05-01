from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .trust_framework import TrustFramework

class TrustFrameworkAPI:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = TrustFramework()
        return cls._instance

class InputView(APIView):
    """
    API endpoint for inputting text to the trust framework
    """
    def post(self, request):
        try:
            text = request.data.get('text')
            if not text:
                return Response(
                    {'error': 'Text is required'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )

            framework = TrustFrameworkAPI.get_instance()
            results = framework.input(text)  # This returns a list
            
            return Response({
                'success': True,
                'results': results,
                'count': len(results)
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response(
                {'error': str(e)}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class RetrieveView(APIView):
    """
    API endpoint for retrieving text from the trust framework
    """
    def post(self, request):
        try:
            query = request.data.get('query')
            if not query:
                return Response(
                    {'error': 'Query is required'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )

            framework = TrustFrameworkAPI.get_instance()
            result = framework.retrieve(query)
            
            return Response(result, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response(
                {'error': str(e)}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class ClearView(APIView):
    """
    API endpoint for clearing all data from the trust framework
    """
    def post(self, request):
        try:
            framework = TrustFrameworkAPI.get_instance()
            framework.clear_all()
            return Response(
                {'message': 'Successfully cleared all data'}, 
                status=status.HTTP_200_OK
            )
        except Exception as e:
            return Response(
                {'error': str(e)}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            ) 