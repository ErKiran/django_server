from keras.preprocessing import image
import numpy as np
from rest_framework import generics
from .serializers import ImageSerializer
from .deeplearning import graph, model
from rest_framework.response import Response


class Predict(generics.CreateAPIView):
    serializer_class = ImageSerializer

    def post(self, request):
        """
            post:
            API to send leaf image and get its health status or disease.
        """
        print(request.data)
        data = ImageSerializer(data=request.data)
        if data.is_valid():
            photo = request.data['photo']
            img = image.load_img(photo, target_size=(224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = img/255

            with graph.as_default():
                prediction = model.predict(img)

            prediction_flatten = prediction.flatten()
          
            return Response({'result': prediction_flatten})
        return Response({'result': data.errors})
