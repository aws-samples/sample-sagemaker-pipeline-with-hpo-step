import boto3
import argparse


# Create CloudWatch client
def create_alarm(pipeline_name, sns_topic_arn):
    cloudwatch = boto3.client('cloudwatch', region_name='us-east-1')
    namespace = 'AWS/Sagemaker/ModelBuildingPipeline'
    sns_topic_arn = sns_topic_arn
    # Create alarm
    try:
        cloudwatch.put_metric_alarm(
            AlarmName='MLPipelineExecitionsSuccess',
            ComparisonOperator='GreaterThanThreshold',
            EvaluationPeriods=1,
            MetricName='ExecutionSucceeded',
            Namespace='AWS/Sagemaker/ModelBuildingPipeline',
            Period=60,
            Statistic='Average',
            Threshold=0,
            ActionsEnabled=True,
            AlarmDescription='Alarm when Pipeline execution success',
            AlarmActions=[sns_topic_arn
                          ],
            Dimensions=[
                {
                    'Name': 'PipelineName',
                    'Value': pipeline_name
                }]
        )
    except  Exception as e:
        print(e)
        print("Error Creating an alarm")


def create_alarm_register_model(pipeline_name, step_name, sns_topic):
    namespace = 'AWS/Sagemaker/ModelBuildingPipeline'
    pipeline_name = pipeline_name
    cloudwatch = boto3.client('cloudwatch', region_name='us-east-1')
    sns_topic_arn = sns_topic
    # Create alarm
    try:
        cloudwatch.put_metric_alarm(
            AlarmName='ModelVersionregistrationSuccess',
            ComparisonOperator='GreaterThanThreshold',
            EvaluationPeriods=1,
            MetricName='StepSucceeded',
            Namespace='AWS/Sagemaker/ModelBuildingPipeline',
            Period=60,
            Statistic='Average',
            Threshold=0,
            ActionsEnabled=True,
            AlarmDescription='Alarm when new Model version registered',
            AlarmActions=[
                sns_topic_arn
            ],
            Dimensions=[
                {
                    'Name': 'PipelineName',
                    'Value': pipeline_name
                },
                {
                    'Name': 'StepName',
                    'Value': step_name
                },
            ]
        )

    except  Exception as e:
        print(e)
        print("Error Creating an alarm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline_name', type=str, default='default')
    parser.add_argument('--sns_topic', type=str, default='default')
    parser.add_argument('--step_name', type=str, default='default')

    args, _ = parser.parse_known_args()
    pipeline_name = args.pipeline_name
    sns_topic = args.sns_topic
    step_name = args.step_name
    print(pipeline_name, sns_topic, step_name)

    create_alarm(pipeline_name, sns_topic)
    create_alarm_register_model(pipeline_name, step_name, sns_topic)

