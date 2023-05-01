#version 450

layout (location = 0) in vec3 position;
layout (location = 1) in vec2 tex_coord;
layout (location = 2) in vec4 tbnquat;

layout( set = 0, binding = 0 ) uniform UScene
{
    mat4 M;
    mat4 V;
    mat4 P;
    vec4 camPos;
} uScene;

layout( location = 0 ) out vec2 v2f_tc;
layout( location = 1 ) out vec4 pos_world;
layout( location = 2 ) out mat3 tbn;

// const float sqrt2 = 1.4142135623730951;
const float sqrt2_half = 0.7071067811865475;

mat3 quaternionToRotationMatrix(vec4 q) {
    float qx2 = q.x * q.x;
    float qy2 = q.y * q.y;
    float qz2 = q.z * q.z;
    float qxqy = q.x * q.y;
    float qxqz = q.x * q.z;
    float qxqw = q.x * q.w;
    float qyqz = q.y * q.z;
    float qyqw = q.y * q.w;
    float qzqw = q.z * q.w;

    return mat3(
        1.0 - 2.0 * (qy2 + qz2), 2.0 * (qxqy + qzqw),     2.0 * (qxqz - qyqw),
        2.0 * (qxqy - qzqw),     1.0 - 2.0 * (qx2 + qz2), 2.0 * (qyqz + qxqw),
        2.0 * (qxqz + qyqw),     2.0 * (qyqz - qxqw),     1.0 - 2.0 * (qx2 + qy2)
    );
}

void main()
{
    pos_world = uScene.M * vec4( position, 1.f );
    gl_Position = uScene.P * uScene.V *pos_world;
    v2f_tc = tex_coord;

    // decode TBN matrix
    int max_component = int(round(tbnquat.a * 3.0));
    // transform to [-sart(2)/2, sqrt(2)/2] range
    vec3 components = tbnquat.rgb * sqrt2_half;
    float max_component_value = sqrt(1.0f - dot(components, components));
    
    int idx = 0;
    vec4 quat = vec4(0.0);
    for (int i = 0; i < 4; i++) {
        if (max_component != i) {
            quat[i] = components[idx ++];
        } else {
            quat[i] = max_component_value;
        }
    }
    tbn = quaternionToRotationMatrix(quat);
    tbn = mat3(normalize(tbn[0]), normalize(tbn[1]), normalize(tbn[2]));
}
