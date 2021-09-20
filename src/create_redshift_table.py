import time
import argparse
import boto3

def execute_query(client,
                  CLUSTER_IDENTIFIER ,
                  DB_NAME ,
                  DB_USER,
                  query):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_name', type=str, default='default')
    parser.add_argument('--db_user', type=str, default='default')
    parser.add_argument('--cluster_id', type=str, default='default')
    parser.add_argument('--role_arn', type=str, default='default')
    parser.add_argument('--bucket_name', type=str, default='default')
    parser.add_argument('--region', type=str, default='default')

    args, _ = parser.parse_known_args()
    db_name = args.db_name
    db_user = args.db_user
    cluster_id = args.cluster_id
    role_arn = args.role_arn
    bucket_name = args.bucket_name
    region = args.region
    
    print(db_name ,db_user,cluster_id,role_arn, bucket_name , region)
    
    
    table_name = f"{db_name}.public.abalone"
    create_table = f'create table IF NOT EXISTS  {table_name}( sex varchar(2), length decimal(8,2), diameter decimal(8,2), height decimal(8,2), whole_weight decimal(8,2), shucked_weight decimal(8,2), viscera_weight decimal(8,2), shell_weight decimal(8,2), rings decimal(8,2));'
    rs_data_client = boto3.client('redshift-data', region_name = region)
    
    execute_query_response = execute_query(rs_data_client,cluster_id , db_name , db_user, create_table + 'COMMIT;')
    query_result = get_query_results(rs_data_client, execute_query_response['Id'])
    print(execute_query_response)

    ## copy data from s3 into redshift table
    file = f's3://{bucket_name}/abalone/'
    copy = f"copy {table_name} from '{file}' credentials 'aws_iam_role={role_arn}'  delimiter ',' region '{region}';"
    rs_data_client = boto3.client('redshift-data')
    print(copy)
                    
                    
    execute_query_response = execute_query(rs_data_client,cluster_id , db_name , db_user,copy + 'COMMIT;')
    query_result = get_query_results(rs_data_client, execute_query_response['Id'])
    print(query_result)