import re
import time
import dateutil.parser as dp

from skeleton import Skeleton
from utils import load_options
from gesture import GestureRecognizer
from is_msgs.image_pb2 import ObjectAnnotations
from prometheus_client import start_http_server, Gauge
from opencensus.ext.zipkin.trace_exporter import ZipkinExporter
from is_wire.core import Subscription, Channel, Logger, Tracer, AsyncTransport


def span_duration_ms(span):
    """Funtion to measure the time of a span in the Zipkin instrumentation.
    
    Parameters
    ----------
    span: Tracer.span
        Represents a single operation in a trace.

    Returns
    -------
    float
        Total amoung of time that took the span.
    """
    dt = dp.parse(span.end_time) - dp.parse(span.start_time)
    return dt.total_seconds() * 1000.0


def create_exporter(service_name, uri):
    """ Funtion to create the exporter in the Zipkin instrumentation.

    Parameters
    ----------
    service_name: str
        Name of the service the will appear on zipkin.
    uri: str
        Zipkin URI.

    Returns
    -------
    ZipkinExporter
        The OpenCensus Zipkin Trace Exporter is a trace exporter that exports data to Zipkin.
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


def select_skeletons(annotations: ObjectAnnotations,
                     min_keypoints: int = 8,
                     x_range: list = [-1, 1],
                     y_range: list = [-1, 1]):
    """Select skeletons on a region.

    Parameters
    ----------
        annotations: is_msgs.image_pb2.ObjectAnnotations
            annotations with all ObjectAnnotation
        min_keypoints: int
            minimum number of keypoints within the region
        x_range: tuple
            x axis limits
        y_range: tuple
            y axis limits

    Returns
    -------
    None
        if there is no skeleton in the region defined with a minimal amount of keypoints.
    is_msgs.image_pb2.ObjectAnnotation
        First skeleton found on that region defined.
    """
    count = 0
    skeleton = None
    for i in range(len(annotations.objects)):
        for j in range(len(annotations.objects[i].keypoints)):
            if annotations.objects[i].keypoints[
                    j].position.x < x_range[1] and annotations.objects[i].keypoints[
                        j].position.x > x_range[0]:
                if annotations.objects[i].keypoints[
                        j].position.y < y_range[1] and annotations.objects[i].keypoints[
                            j].position.y > y_range[0]:
                    count += 1

                if count == min_keypoints:
                    skeleton = annotations.objects[i]
                    break
        count = 0
        if skeleton is not None:
            break
    return skeleton


def main():

    service_name = 'GestureRecognizer.Recognition'
    log = Logger(name=service_name)

    op = load_options()

    channel = Channel(op.broker_uri)
    log.info('Connected to broker {}', op.broker_uri)

    exporter = create_exporter(service_name=service_name, uri=op.zipkin_uri)

    subscription = Subscription(channel=channel, name=service_name)
    for group_id in list(op.group_ids):
        subscription.subscribe('SkeletonsGrouper.{}.Localization'.format(group_id))

    model = GestureRecognizer("model_gesture1_72.00.pth")
    log.info('Initialize the model')

    unc = Gauge('uncertainty_total', "Uncertainty about predict")
    unc.set(0.0)
    start_http_server(8000)

    buffer = list()
    predict_flag = False

    mean = lambda x: (sum(x) / len(x))

    while True:

        msg = channel.consume()

        tracer = Tracer(exporter, span_context=msg.extract_tracing())
        span = tracer.start_span(name='detection_and_info')

        annotations = msg.unpack(ObjectAnnotations)
        skeleton = select_skeletons(annotations=annotations,
                                    min_keypoints=op.skeletons.min_keypoints,
                                    x_range=op.skeletons.x_range,
                                    y_range=op.skeletons.y_range)
        
        if skeleton is None:
            tracer.end_span()
            continue

        skl = Skeleton(skeleton)
        skl_normalized = skl.normalize()
        pred, prob, uncertainty = model.predict(skl_normalized)
            
        if pred == 0 and predict_flag is False:
            pass

        elif pred != 0 and predict_flag is False:
            initial_time = time.time()
            predict_flag = True
            buffer.append(uncertainty)

        elif pred != 0 and predict_flag is True:
            buffer.append(uncertainty)

        elif pred == 0 and predict_flag is True:
            predict_flag = False
            exec_time = time.time() - initial_time
            if exec_time >= op.exec_time:
                unc.set(mean(buffer))
                log.info ("execution_ms: {}, buffer_mean: {}", (exec_time*1000), mean(buffer))
            buffer = []
            
        
        tracer.end_span()

        info = {
            'prediction': pred,
            'probability': prob,
            'uncertainty': uncertainty,
            'took_ms': {
                'service': round(span_duration_ms(span), 2)
            }
        }
        log.info('{}', str(info).replace("'", '"'))


if __name__ == "__main__":
    main()
