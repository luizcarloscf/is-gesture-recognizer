import re
import time
import dateutil.parser as dp

from skeleton import Skeleton
from utils import load_options
from gesture import GestureRecognizer
from is_msgs.image_pb2 import ObjectAnnotations
from prometheus_client import start_http_server, Gauge
from opencensus.ext.zipkin.trace_exporter import ZipkinExporter
from is_wire.core import Subscription, Message, Channel, Logger, Tracer, AsyncTransport


def span_duration_ms(span):
    """ Funtion to measure the time of a span in the Zipkin instrumentation
    """
    dt = dp.parse(span.end_time) - dp.parse(span.start_time)
    return dt.total_seconds() * 1000.0


def create_exporter(service_name, uri):
    """ Funtion to create the exporter in the Zipkin instrumentation
    """
    log = Logger(name="CreateExporter")
    zipkin_ok = re.match("http:\\/\\/([a-zA-Z0-9\\.]+)(:(\\d+))?", uri)
    if not zipkin_ok:
        log.critical("Invalid zipkin uri \"{}\", expected http://<hostname>:<port>", uri)
    exporter = ZipkinExporter(service_name=service_name,
                              host_name=zipkin_ok.group(1),
                              port=zipkin_ok.group(3),
                              transport=AsyncTransport)
    return exporter


def main():

    # Defining our service
    service_name = 'GestureRecognizer.Recognition'
    log = Logger(name=service_name)

    # Loading options
    op = load_options()

    # Conecting to the broker
    channel = Channel(op.broker_uri)
    log.info('Connected to broker {}', op.broker_uri)

    # creating the Zipking exporter
    exporter = create_exporter(service_name=service_name, uri=op.zipkin_uri)

    # Subcripting on desired topics
    subscription = Subscription(channel=channel, name=service_name)
    for group_id in list(op.group_ids):
        subscription.subscribe('SkeletonsGrouper.{}.Localization'.format(group_id))

    # initialize the Model
    model = GestureRecognizer("model_gesture1_72.00.pth")
    log.info('Initialize the model')

    # metrics for monitoring the system
    unc = Gauge('uncertainty_total', "Uncertainty about predict")
    #std = Gauge('std_total', "standard deviation about predict")

    # default values of the metrics=
    unc.set(0)
    #std.set(0)

    # starting the server
    start_http_server(8000)

    # list and time to take the median
    buffer = list()
    #buffer_std = list()
    predict_flag = False

    # begining the service
    while True:

        # waiting for receive a message
        msg = channel.consume()

        # initialize the Tracer
        tracer = Tracer(exporter, span_context=msg.extract_tracing())
        span = tracer.start_span(name='detection_and_info')
        detection_span = None
        skeleton = None
        # unpack the message

        count = 0
        with tracer.span(name='unpack'):
            annotations = msg.unpack(ObjectAnnotations)
            for i in range(len(annotations.objects)):
                for j in range(len(annotations.objects[i].keypoints)):
                    if annotations.objects[i].keypoints[j].position.x < 0.3 and annotations.objects[
                            i].keypoints[j].position.x > -0.3:
                        if annotations.objects[i].keypoints[
                                j].position.y < 0.3 and annotations.objects[i].keypoints[
                                    j].position.y > -0.3:
                            count += 1

                        if count == 8:
                            skeleton = annotations.objects[i]
                            break
                count = 0

                if skeleton is not None:
                    break

            if skeleton is not None:

                skl = Skeleton(skeleton)
                skl_normalized = skl.normalize()
                #skl_vector = skl_normalized.vectorized()

        # preditic
        with tracer.span(name='detection') as _span:
            if skeleton is not None:
                pred, prob, uncertainty = model.predict(skl_normalized)
                detection_span = _span

        # finish the tracer
        tracer.end_span()

        # update metrics values
        if skeleton is not None:

            if pred == 0 and predict_flag == False:
                pass

            elif pred != 0 and predict_flag == False:
                predict_flag = True
                buffer.append(uncertainty)
                #buffer_std.append(std_dev)

            elif pred != 0 and predict_flag == True:
                buffer.append(uncertainty)
                #buffer_std.append(std_dev)

            elif pred == 0 and predict_flag == True:
                predict_flag = False
                unc.set(sum(buffer) / len(buffer))
                #std.set(sum(buffer_std) / len(buffer_std))
                buffer = []
                #buffer_std = []

            # logging usefull informations
            info = {
                'prediction': pred,
                'probability': prob,
                'uncertainty': uncertainty,
                'standard deviation': 0,
                'took_ms': {
                    'detection': round(span_duration_ms(detection_span), 2),
                    'service': round(span_duration_ms(span), 2)
                }
            }
            log.info('{}', str(info).replace("'", '"'))


if __name__ == "__main__":
    main()