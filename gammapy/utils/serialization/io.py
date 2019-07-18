# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Utilities to import and export models
"""
from astropy.units import Unit, Quantity
from ..scripts import read_yaml

def models_to_dict(components_list, do_cut=False):
    dict_list=[]
    for ks in components_list:
        try:
            tmp_dict={'Name':ks.name}
        except:
            tmp_dict={'Name':ks.__class__.__name__}
        try:
            tmp_dict['ID']=ks.ID
        except:
            pass    
        try:
            if ks.file != '': tmp_dict['File']=ks.file
        except:
            pass 
        if ks.__class__.__name__ == 'SkyModel':
                km=ks.spatial_model
                tmp_dict['SkySpatialModel']={'Type':km.__class__.__name__, \
                                               'parameters':km.parameters.to_dict(do_cut)['parameters']}
                try: 
                    tmp_dict['SkySpatialModel']['File'] = km.file
                except:
                    pass
                
                km=ks.spectral_model
    
                tmp_dict['SpectralModel']={'Type':km.__class__.__name__, \
                                            'parameters':km.parameters.to_dict(do_cut)['parameters']} 
                try :
                    tmp_dict['SpectralModel']['energy'] = {"data": km.energy.data.tolist(),\
                                                            "unit": str(km.energy.unit)}
                    tmp_dict['SpectralModel']['values'] = {"data": km.values.data.tolist(),\
                                                            "unit": str(km.values.unit)}
                except:
                    pass
                    
                try:
                    km=ks.temporal_model
                    tmp_dict['TemporalModel']={'Type':km.__class__.__name__, \
                                                   'parameters':km.parameters.to_dict(do_cut)['parameters']}
                except:
                    pass
        else:
            tmp_dict['Model']={'Type':ks.__class__.__name__, \
                                   'parameters':ks.parameters.to_dict(do_cut)['parameters']}
        dict_list.append(tmp_dict)        
    
    # remove duplicates 
    # typically diffuse model with global parameters repeated in different MapDatasets backgrounds
    seen = set()
    dict_list_unique = []
    for kd in dict_list:
        try:
            t = tuple((kd['Name'],kd['ID']))
            if t not in seen:
                seen.add(t)
                dict_list_unique.append(kd)
        except:        
            dict_list_unique.append(kd)
            
    components_dict={'Components':dict_list_unique}
    return components_dict

def dict_to_models(filemodel): 
    from ..fitting import Parameters
    from ...image import models as spatial
    from ...spectrum import models as spectral
    from ...cube.models import SkyModel
    components_list = read_yaml(filemodel)['Components']    
    models_list = []
    for ks in components_list:
        keys=list(ks.keys())
        if 'SkySpatialModel' in keys and 'SpectralModel' in keys:
            
            item = ks['SkySpatialModel']
            if 'File' in list(item.keys()):
                spatial_model= getattr(spatial,item['Type']).read(item['File'])
                spatial_model.file = item['File']
                spatial_model.parameters = Parameters.from_dict(item)
            else:
                params = {x['name']:x['value']*Unit(x['unit']) for x in item['parameters']}
                spatial_model = getattr(spatial,item['Type'])(**params)
                spatial_model.parameters = Parameters.from_dict(item)
                
            item = ks['SpectralModel']
            if 'energy' in list(item.keys()):
                energy=Quantity(item["energy"]["data"], item["energy"]["unit"])
                values=Quantity(item["values"]["data"], item["values"]["unit"])
                params = {"energy":energy,"values":values}
                spectral_model = getattr(spectral,item['Type'])(**params)
                spectral_model.parameters = Parameters.from_dict(item)
            else:
                params = {x['name']:x['value']*Unit(x['unit']) for x in item['parameters']}
                spectral_model = getattr(spectral,item['Type'])(**params)
                spectral_model.parameters = Parameters.from_dict(item)

            models_list.append(SkyModel(name=ks['Name'], \
                                    spatial_model=spatial_model, \
                                    spectral_model=spectral_model)) 
    return models_list