#version 450

layout( location = 0 ) in vec2 v2f_tc;
layout( location = 1 ) in vec4 pos_world;
layout( location = 2 ) in vec3 normal_world;
layout( location = 3 ) in mat3 tbn;

layout( location = 0) out vec4 oColor;

layout( set = 0, binding = 0 ) uniform UScene
{
    mat4 M;
    mat4 V;
    mat4 P;
    vec4 camPos;
} uScene;

layout( set = 1, binding = 0 ) uniform Light
{
    vec4 pos;
    vec4 color;
} uLight;

layout(push_constant) uniform ShadingBit {
    int shadingBit;
} uShadingBit;

void main()
{
    vec4 color = vec4(0);
    switch(uShadingBit.shadingBit) {
        case 0: {
            // normal
            color = vec4( normalize(normal_world), 1.0 ); 
            break; 
        }
        case 1: {
            // view direction
            color = vec4( normalize(uScene.camPos.xyz - pos_world.xyz), 1.0 ); 
            break; 
        }
        default: {
            // light direction
            color = vec4( normalize(uLight.pos.xyz - pos_world.xyz), 1.0 ); 
            break; 
        }
    };
    // normal as color
    oColor = color;
}
