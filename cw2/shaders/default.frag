#version 450

layout( location = 0 ) in vec2 v2f_tc;
layout( location = 1 ) in vec4 pos_world;
layout( location = 2 ) in vec3 normal_world;
layout( location = 3 ) in vec4 v2f_tang;

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

void main()
{
    vec3 light_dir = normalize(uLight.pos.xyz - pos_world.xyz);
    // normal as color
    oColor = vec4( light_dir, 1.0 );
}
