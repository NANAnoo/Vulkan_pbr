#version 450

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 tex_coord;
layout (location = 3) in vec4 tangent;

layout( set = 0, binding = 0 ) uniform UScene
{
    mat4 M;
    mat4 V;
    mat4 P;
    vec4 camPos;
} uScene;

layout( location = 0 ) out vec2 v2f_tc;
layout( location = 1 ) out vec4 pos_world;
layout( location = 2 ) out vec3 normal_world;
layout( location = 3 ) out mat3 tbn;

void main()
{
    pos_world = uScene.M * vec4( position, 1.f );
    normal_world = normalize(mat3( uScene.M ) * normal);
    gl_Position = uScene.P * uScene.V *pos_world;
    v2f_tc = tex_coord;

    vec3 T = normalize( mat3( uScene.M ) * tangent.xyz );
    vec3 B = normalize( cross( normal_world, T )) * tangent.w; 
    tbn = mat3( T, B, normal_world );
}
