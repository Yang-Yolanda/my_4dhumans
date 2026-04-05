import detectron2.data.transforms as T
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, instantiate
from detectron2.data import MetadataCatalog
from detectron2.layers import batched_nms
from omegaconf import OmegaConf


class DefaultPredictor_Lazy:
    """Create a simple end-to-end predictor with the given config that runs on single device for a
    single input image.

    Compared to using the model directly, this class does the following additions:

    1. Load checkpoint from the weights specified in config (cfg.MODEL.WEIGHTS).
    2. Always take BGR image as the input and apply format conversion internally.
    3. Apply resizing defined by the config (`cfg.INPUT.{MIN,MAX}_SIZE_TEST`).
    4. Take one input image and produce a single output, instead of a batch.

    This is meant for simple demo purposes, so it does the above steps automatically.
    This is not meant for benchmarks or running complicated inference logic.
    If you'd like to do anything more complicated, please refer to its source code as
    examples to build and use the model manually.

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            test dataset name in the config.


    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
        """
        Args:
            cfg: a yacs CfgNode or a omegaconf dict object.
        """
        if isinstance(cfg, CfgNode):
            self.cfg = cfg.clone()  # cfg can be modified by model
            self.model = build_model(self.cfg)  # noqa: F821
            if len(cfg.DATASETS.TEST):
                test_dataset = cfg.DATASETS.TEST[0]

            checkpointer = DetectionCheckpointer(self.model)
            checkpointer.load(cfg.MODEL.WEIGHTS)

            self.aug = T.ResizeShortestEdge(
                [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
            )

            self.input_format = cfg.INPUT.FORMAT
        else:  # new LazyConfig
            self.cfg = cfg
            self.model = instantiate(cfg.model)
            test_dataset = OmegaConf.select(cfg, "dataloader.test.dataset.names", default=None)
            if isinstance(test_dataset, (list, tuple)):
                test_dataset = test_dataset[0]

            checkpointer = DetectionCheckpointer(self.model)
            checkpointer.load(OmegaConf.select(cfg, "train.init_checkpoint", default=""))

            mapper = instantiate(cfg.dataloader.test.mapper)
            self.aug = mapper.augmentations
            self.input_format = mapper.image_format

        self.model.eval().cuda()
        if test_dataset:
            self.metadata = MetadataCatalog.get(test_dataset)
        assert self.input_format in ["RGB", "BGR"], self.input_format


    def __call__(self, original_image):
        with torch.no_grad():
            if self.input_format == "RGB":
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug(T.AugInput(original_image)).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            
            # --- 实验开始 ---
            predictions = self.model([inputs])[0]
            
            if "instances" in predictions:
                instances = predictions["instances"]
                num_before = len(instances) # 记录原始数量
                
                # 1. 打印原始分数（看看是不是很多低分垃圾）
                # 只打印前10个，避免刷屏
                raw_scores = instances.scores.tolist()[:10]
                
                # === 过滤步骤 A: 基础阈值 (0.5) ===
                keep = instances.scores > 0.9
                instances = instances[keep]
                num_after_thresh = len(instances)

                # === 过滤步骤 B: 只留人 (Class 0) ===
                if instances.has("pred_classes"):
                    keep = instances.pred_classes == 0
                    instances = instances[keep]
                
                # === 过滤步骤 C: NMS (去重) ===
                keep = batched_nms(
                    instances.pred_boxes.tensor, 
                    instances.scores, 
                    instances.pred_classes, 
                    0.3 # IoU 阈值
                )
                instances = instances[keep]
                num_final = len(instances)

                # 2. 只有当数量发生变化时，才打印日志（证明过滤生效了）
                if num_before != num_final:
                    print(f"\n[实验数据] 原始框数: {num_before} -> 阈值后: {num_after_thresh} -> NMS后(最终): {num_final}")
                    print(f"   >>> 原始分数分布(前10个): {[round(s,2) for s in raw_scores]}")
                    if num_final > 5:
                         print(f"   ⚠️ 警告：过滤后依然有 {num_final} 个人，可能阈值 0.5 还是太低！")

                predictions["instances"] = instances
            
            return predictions