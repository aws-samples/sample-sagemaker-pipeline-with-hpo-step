import json
import boto3
import time
import datetime
import os
DB_NAME = os.environ.get('DB_NAME')
DB_USER = os.environ.get('DB_USER')
CLUSTER_IDENTIFIER = os.environ.get('CLUSTER_IDENTIFIER')
ROLE_ARN = os.environ.get('ROLE_ARN')
PIPELINE_NAME = os.environ.get('PIPELINE_NAME')
BUCKET_NAME = os.environ.get('BUCKET_NAME')
REGION = os.environ.get('REGION')
table_name = f"{DB_NAME}.public.abalone"

def execute_query(client, query):
    response = client.execute_statement(
        ClusterIdentifier=CLUSTER_IDENTIFIER,
        Database=DB_NAME,
        DbUser=DB_USER,
        Sql=query,
        StatementName='city_finder'
    )
    return response

def query_is_finished(describe_statement_response):
    s = describe_statement_response["Status"]
    if s == "FINISHED":
        return True
    elif s == "ABORTED":
        raise Exception('query aborted', describe_statement_response)
    elif s == "FAILED":
        raise Exception('query failed', describe_statement_response)
    else:
        return False

def get_query_results(client, query_id):
    poll = True
    response = None
    while poll == True:
        response = client.describe_statement(Id=query_id)
        if query_is_finished(response):
            return "FINISHED "
        time.sleep


def lambda_handler(event, context):
    # TODO implement
    print(boto3.__version__)
    ## read the latest data from redshift
    rs_data_client = boto3.client('redshift-data')
    location = f's3://{BUCKET_NAME}/abalone-training={datetime.datetime.now().strftime("%m%d%Y%H%M%S")}/'
    unload_query = f"unload ('select * from {table_name}') to '{location}' iam_role '{ROLE_ARN}' DELIMITER AS ',' PARALLEL OFF ; "
    execute_query_response = execute_query(rs_data_client, unload_query + 'COMMIT;')
    query_result = get_query_results(rs_data_client, execute_query_response['Id'])
    print(query_result)

    ## Start the pipeline
    sm_client = boto3.client("sagemaker", REGION)
    param = {'Name': 'InputDataUrl',
             'Value': location}
    parameters = [param]
    response = sm_client.start_pipeline_execution(
        PipelineName=PIPELINE_NAME,
        PipelineExecutionDisplayName=PIPELINE_NAME + datetime.datetime.now().strftime("%d-%m-%Y-%H%M%S"),
        PipelineParameters=parameters)
    print('Pipeline started')
    print(unload_query)
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }