#pragma once

#include "RenderProgram.hpp"
#include "baked_model.hpp"
#include "VkVertex.hpp"
#include <functional>
#include <memory>
#include <unordered_set>

namespace {
    struct RenderingMesh {
        std::uint32_t materialId = 0;
        std::unique_ptr<VkVBO> positions = nullptr;
        std::unique_ptr<VkVBO> normals = nullptr;
        std::unique_ptr<VkVBO> texcoords = nullptr;
        std::unique_ptr<VkVBO> tangents = nullptr;
        std::unique_ptr<VkVBO> tbnquads = nullptr;

        std::unique_ptr<VkIBO> indices = nullptr;

        RenderingMesh() noexcept = default;
        // disable copy
        RenderingMesh(const RenderingMesh&) = delete;
        RenderingMesh operator=(const RenderingMesh&) = delete;

        // enable move
        RenderingMesh(RenderingMesh&& other) noexcept {
            positions = std::move(other.positions);
            normals = std::move(other.normals);
            texcoords = std::move(other.texcoords);
            indices = std::move(other.indices);
            tangents = std::move(other.tangents);
            tbnquads = std::move(other.tbnquads);
        }
        RenderingMesh& operator=(RenderingMesh&& other) noexcept {
            positions = std::move(other.positions);
            normals = std::move(other.normals);
            texcoords = std::move(other.texcoords);
            indices = std::move(other.indices);
            tangents = std::move(other.tangents);
            tbnquads = std::move(other.tbnquads);
            return *this;
        }
    };

    class RenderingModel {
    public:
        enum MaterialType {
            Base = 0, // pbr base color
            AlphaMasked = 1 << 0,
            NormalMapped = 1 << 1,
            All = AlphaMasked | NormalMapped
        };

        enum DescSetType {
            Normal = 0,
            AlphaMask = 1,
            Full = 3,
            NormalDebug = 4,
        };
        RenderingModel() noexcept = default;
        // disable copy
        RenderingModel(const RenderingModel&) = delete;
        RenderingModel operator=(const RenderingModel&) = delete;

        // enable move
        RenderingModel(RenderingModel&& other) noexcept {
            meshes_by_material = std::move(other.meshes_by_material);
        }
        RenderingModel& operator=(RenderingModel&& other) noexcept {
            meshes_by_material = std::move(other.meshes_by_material);
            return *this;
        }

        void load(
            lut::VulkanContext const& aContext, 
            lut::Allocator const& aAllocator,
            const BakedModel& model) 
        {
            texInfos = model.textures;
            materials = model.materials;
            std::unordered_map<std::string, VkFormat> formats;
            std::unordered_map<std::string, int> channels;
            for (size_t i = 0; i < model.meshes.size(); ++i) {
                auto& mesh = model.meshes[i];
                auto renderingMesh = RenderingMesh();

                renderingMesh.materialId = mesh.materialId;
                // load mesh vertex data
                renderingMesh.positions = std::make_unique<VkVBO>(
                    aContext, aAllocator,
                    mesh.positions.size() * sizeof(glm::vec3),
                    (void *)(mesh.positions.data())
                );
                renderingMesh.normals = std::make_unique<VkVBO>(
                    aContext, aAllocator,
                    mesh.normals.size() * sizeof(glm::vec3),
                    (void *)(mesh.normals.data())
                );
                renderingMesh.texcoords = std::make_unique<VkVBO>(
                    aContext, aAllocator,
                    mesh.texcoords.size() * sizeof(glm::vec2),
                    (void *)(mesh.texcoords.data())
                );
                renderingMesh.indices = std::make_unique<VkIBO>(
                    aContext, aAllocator,
                    mesh.indices.size() * sizeof(uint32_t),
                    mesh.indices.size(), 
                    (void *)(mesh.indices.data())
                );
                renderingMesh.tangents = std::make_unique<VkVBO>(
                    aContext, aAllocator,
                    mesh.tangents.size() * sizeof(glm::vec4),
                    (void *)(mesh.tangents.data())
                );
                renderingMesh.tbnquads = std::make_unique<VkVBO>(
                    aContext, aAllocator,
                    mesh.tbnquads.size() * sizeof(std::uint32_t),
                    (void *)(mesh.tbnquads.data())
                );
                auto &mat = materials[mesh.materialId];
                auto type = MaterialType::Base;
                if (mat.alphaMaskTextureId != 0xffffffff) {
                    type = MaterialType(type | MaterialType::AlphaMasked);
                }
                if (mat.normalMapTextureId != 0xffffffff) {
                    type = MaterialType(type | MaterialType::NormalMapped);
                }
                // arrange mesh by material id
                meshes_by_material[type][mesh.materialId].push_back(std::move(renderingMesh));
                
                // set format and channel of each texture by path
                formats[model.textures[mat.baseColorTextureId].path] = VK_FORMAT_R8G8B8A8_SRGB;
                channels[model.textures[mat.baseColorTextureId].path] = 4;

                formats[model.textures[mat.roughnessTextureId].path] = VK_FORMAT_R8_UNORM;
                channels[model.textures[mat.roughnessTextureId].path] = 1;

                formats[model.textures[mat.metalnessTextureId].path] = VK_FORMAT_R8_UNORM;
                channels[model.textures[mat.metalnessTextureId].path] = 1;

                if (mat.normalMapTextureId != 0xffffffff) {
                    // VK_FORMAT_R8G8B8_UNORM not support on my laptop
                    formats[model.textures[mat.normalMapTextureId].path] = VK_FORMAT_R8G8B8A8_UNORM;
                    channels[model.textures[mat.normalMapTextureId].path] = 4;
                } 
            }

            auto TexDescLayourHelper = [&](unsigned int tex_count) {
                std::vector<VkDescriptorSetLayoutBinding> bindings(tex_count);
                for (unsigned int i = 0; i < tex_count; i ++) {
                    bindings[i].binding = i; // bind texture sampler 0
                    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    bindings[i].descriptorCount = 1; 
                    bindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT|VK_SHADER_STAGE_GEOMETRY_BIT; 
                }
                
                VkDescriptorSetLayoutCreateInfo layoutInfo{};
                layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
                layoutInfo.bindingCount = uint32_t(bindings.size());
                layoutInfo.pBindings = bindings.data();

                VkDescriptorSetLayout layout = VK_NULL_HANDLE;
                if( auto const res = vkCreateDescriptorSetLayout( 
                        aContext.device, 
                        &layoutInfo,
                        nullptr, 
                        &layout ); 
                    VK_SUCCESS != res )
                {
                    throw lut::Error( "Unable to create descriptor set layout\n" 
                        "vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());
                }
                return lut::DescriptorSetLayout( aContext.device, layout ); 
            };

            std::vector<std::string> paths;
            for (auto& tex : texInfos) {
                paths.push_back(tex.path);
            }
            labutils::CommandPool tempPool = labutils::create_command_pool(aContext, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
            auto res = labutils::loadTextures(aContext, tempPool.handle, aAllocator, paths, formats, channels);
            // create img view for each img
            for (auto& [path, img] : res) {
                auto texView = lut::create_image_view_texture2d(aContext, img.image, formats[path]);
                PackedTexture tex{
                    std::move(img),
                    std::move(texView)
                };
                allTextures.insert({
                    path,
                    std::move(tex)
                });
            }
            pbrBaseLayout = TexDescLayourHelper(3);
            pbrHalfLayout = TexDescLayourHelper(4);
            pbrFullLayout = TexDescLayourHelper(5);
            debugLayout = TexDescLayourHelper(1);

            uploadScope(aContext, [&aContext, &aAllocator, this](VkCommandBuffer uploadCmd) {
                // load dummy texture
                dummyTexture = lut::load_dummy_texture(aContext, uploadCmd, aAllocator);
                auto const& [img, _] = dummyTexture;
                dummyTextureView = lut::create_image_view_texture2d(aContext, img.image, VK_FORMAT_R8G8B8A8_SRGB);
            });
        }

        void upload(VkCommandBuffer uploadCmd) {
            for (auto& [_, mesh_map] : meshes_by_material) {
                for (auto &[_, meshes] : mesh_map) {
                    for (auto& mesh : meshes) {
                        mesh.positions->upload(uploadCmd);
                        mesh.normals->upload(uploadCmd);
                        mesh.texcoords->upload(uploadCmd);
                        mesh.tangents->upload(uploadCmd);
                        mesh.tbnquads->upload(uploadCmd);
                        mesh.indices->upload(uploadCmd);
                    }
                }
            }
        }

        PipeLineGenerator bindPipeLine(PipeLineGenerator aGenerator) {
            aGenerator
            // positions
            .addVertexInfo(0, 0, sizeof(float) * 3, VK_FORMAT_R32G32B32_SFLOAT)
            // normals
            .addVertexInfo(1, 1, sizeof(float) * 3, VK_FORMAT_R32G32B32_SFLOAT)
            // texcoords
            .addVertexInfo(2, 2, sizeof(float) * 2, VK_FORMAT_R32G32_SFLOAT)
            // tangents
            .addVertexInfo(3, 3, sizeof(float) * 4, VK_FORMAT_R32G32B32A32_SFLOAT);
            return aGenerator;
        }

        PipeLineGenerator bindTbnCompressedPipeLine(PipeLineGenerator aGenerator) {
            aGenerator
            // positions
            .addVertexInfo(0, 0, sizeof(float) * 3, VK_FORMAT_R32G32B32_SFLOAT)
            // texcoords
            .addVertexInfo(1, 1, sizeof(float) * 2, VK_FORMAT_R32G32_SFLOAT)
            // tbnquads
            .addVertexInfo(2, 2, sizeof(uint32_t), VK_FORMAT_A2R10G10B10_UNORM_PACK32);
            return aGenerator;
        }

        void onDraw(MaterialType aType, DescSetType which_set, VkCommandBuffer cmd, const std::function<void(VkDescriptorSet set)> &aSetCallback) {
            for (auto& [mat_id, meshes] : meshes_by_material[aType]) {
                if (which_set == DescSetType::Normal)
                    aSetCallback(material_basic_sets[mat_id]);
                else if (which_set == DescSetType::AlphaMask)
                    aSetCallback(material_alpha_test_sets[mat_id]);
                else if (which_set == DescSetType::Full)
                    aSetCallback(material_full_sets[mat_id]);
                else if (which_set == DescSetType::NormalDebug)
                    aSetCallback(material_normdebug_sets[mat_id]);
                for (auto& mesh : meshes) {
                    VkBuffer buffers[4] = { mesh.positions->get(), mesh.normals->get(), mesh.texcoords->get(), mesh.tangents->get()};
                    VkDeviceSize offsets[4] = { 0, 0, 0 , 0};
                    vkCmdBindVertexBuffers(cmd, 0, 4, buffers, offsets);
                    mesh.indices->bind(cmd);
                    mesh.indices->draw(cmd);
                }
            }
        }

        void tbnCompressedDraw(VkCommandBuffer cmd, const std::function<void(VkDescriptorSet set)> &aSetCallback) {
            for (auto &[_, meshes] : meshes_by_material) {
                for (auto &[mat_id, sub_meshes] : meshes) {
                    aSetCallback(material_dummy_sets[mat_id]);
                    for (auto& mesh : sub_meshes) {
                        VkBuffer buffers[3] = { mesh.positions->get(), mesh.texcoords->get(), mesh.tbnquads->get() };
                        VkDeviceSize offsets[3] = { 0, 0, 0 };
                        vkCmdBindVertexBuffers(cmd, 0, 3, buffers, offsets);
                        mesh.indices->bind(cmd);
                        mesh.indices->draw(cmd);
                    }
                }
            }
        }

        void createSetsWith(
            lut::VulkanContext const& aContext,
            VkDescriptorPool dPool,
            VkSampler aSampler)
        {
            // create material descriptor sets
            std::vector<VkDescriptorImageInfo> imgInfos;
            std::vector<VkWriteDescriptorSet> texWriteSets;

            VkDescriptorImageInfo dummyImgInfo = {
                aSampler,
                dummyTextureView.handle,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            };
            
            // have to reserve a enough space
            // since when vector increase
            // stl will move the old array buffer to the new buffer
            imgInfos.reserve(texInfos.size());
            // load & upload all textures
            for (auto &info :texInfos) {
                auto &tex = allTextures[info.path];
                imgInfos.push_back({
                    aSampler,
                    tex.view.handle,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                });
            }

            auto CreateWriteDesc = [&](VkDescriptorSet set, VkDescriptorImageInfo *imgInfo, unsigned int count) {
                VkWriteDescriptorSet desc{};
                desc.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                desc.dstSet = set; 
                desc.dstBinding = 0; //
                desc.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                desc.descriptorCount = count; 
                desc.pImageInfo = imgInfo; 
                texWriteSets.emplace_back(desc);
            };

            std::vector<std::vector<VkDescriptorImageInfo>> mat_imgInfos(materials.size());
            std::vector<std::vector<VkDescriptorImageInfo>> dummy_imgInfos(materials.size());
            for (unsigned int i = 0; i < materials.size(); i ++) {
                auto &mat = materials[i];
                auto &mat_imgInfo = mat_imgInfos[i];
                mat_imgInfo.reserve(5);
                auto &dummy_imgInfo = dummy_imgInfos[i];
                for (auto &idx : {
                    mat.baseColorTextureId, 
                    mat.metalnessTextureId, 
                    mat.roughnessTextureId,
                    mat.alphaMaskTextureId,
                    mat.normalMapTextureId
                }) {
                    if (idx != 0xffffffff)
                        mat_imgInfo.emplace_back(imgInfos[idx]);
                    dummy_imgInfo.emplace_back(idx != 0xffffffff ? imgInfos[idx] : dummyImgInfo);
                }
                auto basic_set = lut::alloc_desc_set(aContext, dPool, pbrBaseLayout.handle);
                material_basic_sets.push_back(basic_set);
                auto dummy_set = lut::alloc_desc_set(aContext, dPool, pbrFullLayout.handle);
                material_dummy_sets.push_back(dummy_set);
                CreateWriteDesc(dummy_set, dummy_imgInfo.data(), 5);
                
                if (mat.alphaMaskTextureId != 0xffffffff) {
                    auto alpha_test_set = lut::alloc_desc_set(aContext, dPool, pbrHalfLayout.handle);
                    material_alpha_test_sets.push_back(alpha_test_set);
                    CreateWriteDesc(alpha_test_set, mat_imgInfo.data(), 4);
                    if ( mat.normalMapTextureId != 0xffffffff) {
                        auto full_set = lut::alloc_desc_set(aContext, dPool, pbrFullLayout.handle);
                        material_full_sets.push_back(full_set);
                        CreateWriteDesc(full_set, mat_imgInfo.data(), 5);
                    } else {
                        material_full_sets.push_back(alpha_test_set);
                    }
                } else {
                    if ( mat.normalMapTextureId != 0xffffffff) {
                        auto normal_map_set = lut::alloc_desc_set(aContext, dPool, pbrHalfLayout.handle);
                        material_full_sets.push_back(normal_map_set);
                        CreateWriteDesc(normal_map_set, mat_imgInfo.data(), 4);
                    } else {
                        material_full_sets.push_back(basic_set);
                    }
                    material_alpha_test_sets.push_back(basic_set);
                }
                CreateWriteDesc(basic_set, mat_imgInfo.data(), 3);

                if (mat.normalMapTextureId != 0xffffffff) {
                    auto debug_set = lut::alloc_desc_set(aContext, dPool, debugLayout.handle);
                    material_normdebug_sets[i] = debug_set;
                    CreateWriteDesc(debug_set, &imgInfos[mat.normalMapTextureId], 1);
                }
            }

            // update descriptor set
            vkUpdateDescriptorSets( aContext.device, static_cast<std::uint32_t>(texWriteSets.size()), texWriteSets.data(), 0, nullptr );
        }

        static void uploadScope(lut::VulkanContext const& aContext, const std::function<void(VkCommandBuffer)> &cb) {
            labutils::CommandPool tempPool = labutils::create_command_pool(aContext, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
            // create upload cmd buffer, fence
            labutils::Fence uploadComplete = labutils::create_fence(aContext);
            VkCommandBuffer uploadCmd = labutils::alloc_command_buffer(aContext, tempPool.handle);

            // begin cmd
            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = 0;
            beginInfo.pInheritanceInfo = nullptr;

            if (auto const res = vkBeginCommandBuffer(uploadCmd, &beginInfo);
                VK_SUCCESS != res) {
                throw lut::Error("Beginning command buffer recording\n"
                    "vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());
            }

            // record upload cmd
            cb(uploadCmd);

            // end cmd
            if (auto const res = vkEndCommandBuffer(uploadCmd); VK_SUCCESS != res) {
                throw lut::Error("Ending command buffer recording\n"
                    "vkEndCommandBuffer() returned %s", lut::to_string(res).c_str());
            }
            // submit upload cmd
            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &uploadCmd;

            if (auto const res = vkQueueSubmit(
                aContext.graphicsQueue, 
                1, &submitInfo, 
                uploadComplete.handle);
                VK_SUCCESS != res) {
                throw lut::Error( "Submitting commands\n" 
                    "vkQueueSubmit() returned %s", lut::to_string(res).c_str());
            }

            // Wait for commands to finish before we destroy the temporary resources
            if( auto const res = vkWaitForFences( aContext.device, 1, &uploadComplete.handle,
                VK_TRUE, std::numeric_limits<std::uint64_t>::max() ); VK_SUCCESS != res ) {
                throw lut::Error( "Waiting for upload to complete\n" 
                    "vkWaitForFences() returned %s", lut::to_string(res).c_str());
            }
            vkFreeCommandBuffers( aContext.device, tempPool.handle, 1, &uploadCmd );
        }

        lut::DescriptorSetLayout pbrBaseLayout; // base color + metalness + roughness
        lut::DescriptorSetLayout pbrHalfLayout; // alpha mask or normal map
        lut::DescriptorSetLayout pbrFullLayout;  // alpha mask + normal map
        lut::DescriptorSetLayout debugLayout; // normal map debug

    private:
    	struct PackedTexture {
            labutils::Image img{};
            labutils::ImageView view{};
        };

        // texture data map
        std::unordered_map<std::string, PackedTexture> allTextures;

        std::tuple<lut::Image, lut::Buffer> dummyTexture;
        labutils::ImageView dummyTextureView;
        
        std::vector<VkDescriptorSet> material_basic_sets;
        std::vector<VkDescriptorSet> material_alpha_test_sets;
        std::vector<VkDescriptorSet> material_full_sets;
        std::vector<VkDescriptorSet> material_dummy_sets;
        std::unordered_map<unsigned int, VkDescriptorSet> material_normdebug_sets;

        std::vector<BakedTextureInfo> texInfos;
        std::unordered_map<MaterialType, std::unordered_map<unsigned int, std::vector<RenderingMesh>>> meshes_by_material;
        std::vector<BakedMaterialInfo> materials;
    };
}