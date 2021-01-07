if __name__ == '__main__':
    import sys
    import time     
    from distributed import Client
    
    fname = sys.argv[1]
    dask_addr = sys.argv[2]

    # get dask client and sleep so it has time to set up
    cli = Client(dask_addr)

    # upload files so workers can import
#     time.sleep(2)
    cli.upload_file("data.py")
    cli.upload_file("hingeloss.py")
    cli.upload_file("model.py")
    cli.upload_file("grapher.py")
    cli.upload_file("experiment.py")
#     time.sleep(2)
    
    from experiment import Experiment
    
    e = Experiment(fname)
    future = cli.submit(e.run)
    future = cli.gather(future)
    print(future)
    


    