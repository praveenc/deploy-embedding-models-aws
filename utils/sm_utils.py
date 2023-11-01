from sagemaker.predictor import Predictor
from sagemaker.huggingface.model import HuggingFaceModel
import boto3
from rich import print
from datetime import datetime, timedelta
from typing import Dict
from sagemaker.serverless import ServerlessInferenceConfig

PYTHON_VER = "py310"
TRANSFORMERS_VER = "4.28.1"
PYTORCH_VER = "2.0.0"


# function to get huggingface inference container uri
def get_hugging_face_image_uri(
    region: str = "us-west-2",
    transformers_version: str = TRANSFORMERS_VER,
    pytorch_version: str = PYTORCH_VER,
    py_version: str = PYTHON_VER,
    device: str = "cpu"
):
    """
    Get HuggingFace Inference Container uri
    Refer here for latest version info:
    https://github.com/aws/deep-learning-containers/blob/master/available_images.md#huggingface-inference-containers
    """
    account_id_map = {
        "us-east-1": "763104351884",
        "us-east-2": "763104351884",
        "us-west-1": "763104351884",
        "us-west-2": "763104351884",
        "af-south-1": "626614931356",
        "ap-east-1": "871362719292",
        "ap-south-1": "763104351884",
        "ap-south-2": "772153158452",
        "ap-northeast-3": "364406365360",
        "ap-northeast-2": "763104351884",
        "ap-southeast-1": "763104351884",
        "ap-southeast-2": "763104351884",
        "ap-southeast-3": "907027046896",
        "ap-southeast-4": "457447274322",
        "ap-northeast-1": "763104351884",
        "ca-central-1": "763104351884",
        "eu-central-1": "763104351884",
        "eu-central-2": "380420809688",
        "eu-west-1": "763104351884",
        "eu-west-2": "763104351884",
        "eu-south-1": "692866216735",
        "eu-south-2": "503227376785",
        "eu-west-3": "763104351884",
        "eu-north-1": "763104351884",
        "il-central-1": "780543022126",
        "me-south-1": "217643126080",
        "me-central-1": "914824155844",
        "sa-east-1": "763104351884",
        "cn-north-1": "727897471807",
        "cn-northwest-1": "727897471807",
    }

    if region not in account_id_map.keys():
        raise ("UNSUPPORTED REGION")

    base = "amazonaws.com.cn" if region.startswith("cn-") else "amazonaws.com"
    account_id = account_id_map[region]
    hf_image_uri = (
        f"{account_id}.dkr.ecr.{region}.{base}/huggingface-pytorch-inference:"
        f"{pytorch_version}-transformers{transformers_version}-{device}"
        f"-{py_version}-ubuntu20.04"
    )
    return hf_image_uri


# function to create and deploy model to Amazon SageMaker HuggingFace DLC
def create_deploy_huggingface_model(
    model_name: str,
    model_s3uri: str,
    role: str,
    instance_type: str = None,
    transformers_version: str = TRANSFORMERS_VER,
    py_version: str = PYTHON_VER,
    pytorch_version: str = PYTORCH_VER,
    serverless_config: ServerlessInferenceConfig = None,
    env: Dict = None,
    wait: bool = False
):
    """
    Refer to https://github.com/aws/deep-learning-containers/blob/master/available_images.md#huggingface-inference-containers
    for all supported version numbers.
    """
    endpoint_name = model_name
    try:
        # create model
        print(f"Creating model: {model_name}")
        hf_model = HuggingFaceModel(
            name=model_name,
            model_data=model_s3uri,
            role=role,
            transformers_version=transformers_version,
            py_version=py_version,
            pytorch_version=pytorch_version,
            predictor_cls=Predictor,
            env=env,
        )
    except Exception as e:
        print(f"Error creating model: {e}")
        raise e

    print(
        f"Deploying to endpoint:[b green]{endpoint_name}[/b green] with \n"
        f"\nenv=[i]{env}[/i]"
    )

    # deploy model
    if serverless_config is not None:
        print(f"Deploying {model_name} to serverless endpoint")
        predictor = hf_model.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            serverless_inference_config=serverless_config,
            wait=wait
        )
    else:
        print(f"Deploying {model_name} to {instance_type} real-time endpoint")
        predictor = hf_model.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            wait=wait,
            container_startup_health_check_timeout=600,
        )
    return predictor


def get_endpoint_status(endpoint_name: str):
    sm_client = boto3.client("sagemaker")
    status = sm_client.describe_endpoint(EndpointName=endpoint_name)["EndpointStatus"]
    print(
          f"[b]Endpoint:[/b] [b green]{endpoint_name}[/b green] | "
          f"[b]Status:[/b] [i bright_red]{status}[/i bright_red]"
    )

    # Get the waiter object
    waiter = sm_client.get_waiter("endpoint_in_service")
    # Apply the waiter on the endpoint
    waiter.wait(EndpointName=endpoint_name)

    # Get endpoint status using describe endpoint
    status = sm_client.describe_endpoint(EndpointName=endpoint_name)["EndpointStatus"]
    print(
          f"[b]Endpoint:[/b] [b green]{endpoint_name}[/b green] | "
          f"[b]Status:[/b] [i magenta3]{status}[/i magenta3] ✅"
    )


def print_cloudwatch_logs(endpoint_name: str):
    logs_client = boto3.client('logs')
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=1)

    log_group_name = f'/aws/sagemaker/Endpoints/{endpoint_name}'
    log_streams = logs_client.describe_log_streams(logGroupName=log_group_name)
    log_stream_name = log_streams['logStreams'][0]['logStreamName']

    # Retrieve the logs
    logs = logs_client.get_log_events(
        logGroupName=log_group_name,
        logStreamName=log_stream_name,
        startTime=int(start_time.timestamp() * 1000),
        endTime=int(end_time.timestamp() * 1000)
    )

    # Print the logs
    for event in logs['events']:
        print(f"{datetime.fromtimestamp(event['timestamp'] // 1000)}: {event['message']}")