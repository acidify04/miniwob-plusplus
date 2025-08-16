import pathlib
from gymnasium.envs.registration import register
from miniwob.fields import create_regex_field_extractor

register( 
    id='miniwob/custom-v0', 
    entry_point='miniwob.environment:MiniWoBEnvironment', 
    kwargs={ 'subdomain': 'custom-click', 'base_url': 'file:///home/cau/Documents/miniwob-plusplus/miniwob/html/', 
            'field_extractor': create_regex_field_extractor( r'Click the button\.', ['amount'], ) 
    } 
)