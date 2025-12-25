from rescue.create_dataset import BulkCreate
def simulation(simulate_path, data_path, sample_size, num_samples, pattern, fmt):
    unknown_celltypes=[]
    unknown_celltypes = list(unknown_celltypes)
    bulk_simulator = BulkCreate(sample_size=sample_size,
                                   num_samples=num_samples,
                                   data_path=data_path,
                                   out_path=simulate_path,
                                   pattern=pattern,
                                   unknown_celltypes=unknown_celltypes,
                                   fmt=fmt)
    bulk_simulator.simulate()