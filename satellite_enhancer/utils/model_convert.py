import coremltools


def convert_model_to_coreml(model):

    model = coremltools.converters.keras.convert(model, input_names=['image'], output_names=['output'])

    # your_model.author = 'your name'
    # your_model.short_description = 'Digit Recognition with MNIST'
    # your_model.input_description['image'] = 'Takes as input an image'
    # your_model.output_description['output'] = 'Prediction of Digit'

    model.save('your_model_name.mlmodel')
