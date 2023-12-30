from stablediffusionxl.demo.script_sampling import VERSION2SPECS, prepare, postprocessing
from stablediffusionxl.demo.script_helpers import get_unique_embedder_keys_from_conditioner, init_embedder_options, init_sampling, do_text2img

def Stable_diffusion_XL_text_to_image(prompt, opt):
    state, return_latents, stage2strength, state2, sampler2, finish_denoising, negative_prompt = prepare(opt)
    filter = state.get("filter")
    version_dict = VERSION2SPECS[opt["version"]]
    if opt["use_custom_res"] == False:
        H = version_dict["H"]
        W = version_dict["W"]
    else:
        H = opt["h"]
        W = opt["w"]
    C = 4
    F = opt["f"]
    init_dict = {
        "orig_width": W,
        "orig_height": H,
        "target_width": W,
        "target_height": H,
    }
    value_dict = init_embedder_options(opt, get_unique_embedder_keys_from_conditioner(state["model"].conditioner), init_dict, prompt = prompt, negative_prompt = negative_prompt if version_dict["is_legacy"] else "")
    sampler, num_rows, num_cols = init_sampling(opt = opt, stage2strength = stage2strength)
    num_samples = num_rows * num_cols
    out = do_text2img(state["model"], sampler, value_dict, num_samples, H, W, C, F, force_uc_zero_embeddings = ["txt"] if not version_dict["is_legacy"] else [], return_latents = return_latents, filter = filter)
    return postprocessing(opt, prompt, state, finish_denoising, out, return_latents, state2, sampler2)

if __name__ == "__main__":
    prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k", #Описание
    opt = {
        "add_watermark": False, #Добавлять невидимую вотермарку
        "version": "SDXL-base-1.0", # Выбор модели: "SDXL-base-1.0", "SDXL-base-0.9" (недоступна для коммерческого использования), "SD-2.1", "SD-2.1-768", "SDXL-refiner-0.9" (недоступна для коммерческого использования),  "SDXL-refiner-1.0"
        "use_custom_ckpt": False, #Использовать свои веса для выбранной версии модели
        "custom_ckpt_name": "sd_xl_refiner_1.0.safetensors", #Имя кастомной модели, если выбран "use_custom_ckpt"
        "low_vram_mode": False, #Режим для работы на малом количестве видеопамяти
        "version2SDXL-refiner": True, #Только для версий SDXL-base: загрузить SDXL-refiner как модель для второй стадии обработки. Требует более длительной обработки и больше видеопамяти
        "seed": 42, #Инициализирующее значение (может быть от 0 до 1000000000)
        "negative_prompt": "", #Для всех моделей, кроме SDXL-base: негативное описание
        "refiner": "SDXL-refiner-1.0", #Если "version2SDXL-refiner" выбран, то какую версию модели для второй стадии обработки загрузить: "SDXL-refiner-1.0", "SDXL-refiner-0.9"  (недоступна для коммерческого использования)
        "refinement_strength": 0.15, #Сила вклада обработки на второй стадии (от 0.0 до 1.0)
        "finish_denoising": True, #Завершить удаление шума рафинёром (только для моделей SDXL-base, если включён version2SDXL-refiner)
        "h": 1024, #Высота желаемого изображения (от 64 до 2048, должна быть кратна 64)
        "w": 1024, #Ширина желаемого изображения (от 64 до 2048, должна быть кратна 64)
        "max_dim": pow(8192, 2), # я не могу генерировать на своей видюхе картинки больше 4096 на 4096
        "c": 4, #Кто знает, что это - напишите
        "f": 8, #Коэффициент понижающей дискретизации, чаще всего 8 или 16 (можно 4, тогда есть риск учетверения, но красиво и жрёт больше видеопамяти)
        "use_custom_res": True, #Использовать установленное самостоятельно для каждой модели разрешение генерации, вместо рекомендованного
        "sampler": "EulerEDMSampler", #обработчик ("EulerEDMSampler", "HeunEDMSampler", "EulerAncestralSampler", "DPMPP2SAncestralSampler", "DPMPP2MSampler", "LinearMultistepSampler")
        "s_churn": 0.0,  #Только для обработчиков "EulerEDMSampler" или "HeunEDMSampler" (от 0.0 до 1.0)
        "s_tmin": 0.0, #Только для обработчиков ("EulerEDMSampler" или "HeunEDMSampler") и "s_churn" > 0, обнуляет сигмы меньше этого значения (от 0.0 до "sigma_max" и < "s_tmax")
        "s_tmax": 999.0,  #Только для обработчиков ("EulerEDMSampler" или "HeunEDMSampler") и "s_churn" > 0, обнуляет сигмы больше этого значения (от "sigma_min" до "sigma_max" и > "s_tmin")
        "s_noise": 1.0, #Только для обработчиков ("EulerEDMSampler" или "HeunEDMSampler" или "EulerAncestralSampler" или "DPMPP2SAncestralSampler") и "s_churn" > 0 (от 0.0)
        "eta": 1.0, #Только для обработчика "EulerAncestralSampler" или "DPMPP2SAncestralSampler" (от 0.0)
        "order": 4, #Только для обработчика "LinearMultistepSampler" (от 1)
        "m_k": 8, #Коэффициент улучшения при постобработке (если активирован version2SDXL-refiner и модель SDXL-base) (понятия не имею от скольки до скольки он может быть, надо тестить)
        "aesthetic_score": 6.0, #Эстетический коэффициент (если активирован version2SDXL-refiner и модель SDXL-base) (понятия не имею от скольки до скольки он может быть, надо тестить)
        "negative_aesthetic_score": 2.5, #Обратный эстетический коэффициент (если активирован version2SDXL-refiner и модель SDXL-base) (понятия не имею от скольки до скольки он может быть, надо тестить)
        "custom_orig_size": False, #Если применён, то меняет размеры входного изображения на "orig_width" и "orig_heigt", иначе оставляет равними размерам желаемого изображения
        "orig_width": 1024, #Ширина входного изображения, если установлен параметр "custom_orig_size" (от 16)
        "orig_heigt": 1024, #Высота входного изображения, если установлен параметр "custom_orig_size" (от 16)
        "crop_coords_top": 0, #Обрезка координат сверху (от 0)
        "crop_coords_left": 0, #Обрезка координат слева (от 0)
        "guider_discretization": "VanillaCFG", #Дискретизатор проводника? ("VanillaCFG", "IdentityGuider")
        "sampling_discretization": "LegacyDDPMDiscretization", #Дискретизатор обработчика ("LegacyDDPMDiscretization", "EDMDiscretization")
        "sigma_min": 0.03, #Только для "EDMDiscretization" дискритизатора обработчика
        "sigma_max": 14.61, #Только для "EDMDiscretization" дискритизатора обработчика
        "rho": 3.0, #Только для "EDMDiscretization" дискритизатора обработчика
        "num_cols": 1, #Количество возвращаемых изображений (от 1 до 10, но, думаю, можно и больше при желании)
        "guidance_scale": 5.0, #Величина guidance (от 0.0 до 100.0)
        "steps": 40, #Количество шагов обработки (от 0 до 1000)
        "use_filter": True, #Использовать фильтр на вотермарки и NSFW
        "verbose": True #Выводить дополнительную отладочную информацию
    }
    r = Stable_diffusion_XL_text_to_image(prompt, opt)
    c = 3
    for rr in r:
        binary_data = rr
        c += 1
        with open("output_" + str(c) + ".png", "wb") as f:
            f.write(binary_data)