import { PointXYZ, Normal } from '@/modules/common/point-types';
import { PointCloud, PointIndices } from '@/modules/common/PointCloud';
import ModelCoefficients from '@/modules/common/ModelCoefficients';
import PassThrough from '@/modules/filters/PassThrough';
import NormalEstimation from '@/modules/features/NormalEstimation';
import SACSegmentationFromNormals from '@/modules/segmentation/SACSegmentationFromNormals';
import ExtractIndices from '@/modules/filters/ExtractIndices';
import { SearchKdTree } from '@/modules/search';
import { SacModelTypes, SacMethodTypes } from '@/modules/sample-consensus/constants';
import { loadPCDData, loadPCDFile } from '@/modules/io';

/**
 * 从点云中分割平面和圆柱体
 * @param inputCloudPath 输入点云的PCD文件路径
 * @param planeOutputPath 输出平面点云的PCD文件路径
 * @param cylinderOutputPath 输出圆柱体点云的PCD文件路径
 * @returns 包含处理结果的对象，包括平面和圆柱体的系数以及各个阶段点云的大小
 */
export async function segmentCylinder(
  inputCloudPath: string,
  planeOutputPath = 'plane.pcd',
  cylinderOutputPath = 'cylinder.pcd',
) {
  // 所有需要的对象

  const pass = new PassThrough<PointXYZ>();
  const ne = new NormalEstimation<PointXYZ, Normal>();
  const seg = new SACSegmentationFromNormals<PointXYZ>();

  const extract = new ExtractIndices<PointXYZ>();
  const extractNormals = new ExtractIndices<Normal>();
  const tree = new SearchKdTree<PointXYZ>();

  // 数据集
  const cloud = new PointCloud<PointXYZ>();
  const cloudFiltered = new PointCloud<PointXYZ>();
  const cloudNormals = new PointCloud<Normal>();
  const cloudFiltered2 = new PointCloud<PointXYZ>();
  const cloudNormals2 = new PointCloud<Normal>();
  const coefficientsPlane = new ModelCoefficients();
  const coefficientsCylinder = new ModelCoefficients();
  const inliersPlane = new PointIndices();
  const inliersCylinder = new PointIndices();

  try {
    // 读取点云数据
    const data = await fetch(inputCloudPath).then((res) => res.arrayBuffer());
    const cloud = loadPCDData(data);
    console.log(`点云包含: ${cloud.size} 个数据点。`);

    // 构建直通滤波器以去除异常值和场景背景
    pass.setInputCloud(cloud);
    pass.setFilterFieldName('z');
    pass.setFilterLimits(0, 1.5);
    pass.filter(cloudFiltered);
    console.log(`过滤后的点云包含: ${cloudFiltered.size} 个数据点。`);

    // 估计点法线
    ne.setSearchMethod(tree);
    ne.setInputCloud(cloudFiltered);
    ne.setKSearch(50);
    ne.compute(cloudNormals);

    // 创建平面模型分割对象并设置所有参数
    seg.setOptimizeCoefficients(true);
    seg.setModelType(SacModelTypes.SACMODEL_NORMAL_PLANE);
    seg.setNormalDistanceWeight(0.1);
    seg.setMethodType(SacMethodTypes.SAC_RANSAC);
    seg.setMaxIterations(100);
    seg.setDistanceThreshold(0.03);
    seg.setInputCloud(cloudFiltered);
    seg.setInputNormals(cloudNormals);

    // 获取平面内点和系数
    seg.segment(inliersPlane, coefficientsPlane);
    console.log(`平面系数: ${coefficientsPlane.values.join(', ')}`);

    // 从输入点云中提取平面内点
    extract.setInputCloud(cloudFiltered);
    extract.setIndices(inliersPlane);
    extract.setNegative(false);

    // 将平面内点写入磁盘
    const cloudPlane = new PointCloud<PointXYZ>();
    extract.filter(cloudPlane);
    console.log(`表示平面组件的点云: ${cloudPlane.size} 个数据点。`);

    // 移除平面内点，提取剩余部分
    extract.setNegative(true);
    extract.filter(cloudFiltered2);
    extractNormals.setNegative(true);
    extractNormals.setInputCloud(cloudNormals);
    extractNormals.setIndices(inliersPlane);
    extractNormals.filter(cloudNormals2);

    // 创建圆柱体分割对象并设置所有参数
    seg.setOptimizeCoefficients(true);
    seg.setModelType(SacModelTypes.SACMODEL_CYLINDER);
    seg.setMethodType(SacMethodTypes.SAC_RANSAC);
    seg.setNormalDistanceWeight(0.1);
    seg.setMaxIterations(10000);
    seg.setDistanceThreshold(0.05);
    seg.setRadiusLimits(0, 0.1);
    seg.setInputCloud(cloudFiltered2);
    seg.setInputNormals(cloudNormals2);

    // 获取圆柱体内点和系数
    seg.segment(inliersCylinder, coefficientsCylinder);
    console.log(`圆柱体系数: ${coefficientsCylinder.values.join(', ')}`);

    // 将圆柱体内点写入磁盘
    extract.setInputCloud(cloudFiltered2);
    extract.setIndices(inliersCylinder);
    extract.setNegative(false);
    const cloudCylinder = new PointCloud<PointXYZ>();
    extract.filter(cloudCylinder);

    if (cloudCylinder.size === 0) {
      console.log('找不到圆柱体组件。');
      return {
        planeCoefficients: coefficientsPlane.values,
        cylinderCoefficients: null,
        originalSize: cloud.size,
        filteredSize: cloudFiltered.size,
        planeSize: cloudPlane.size,
        cylinderSize: 0,
      };
    } else {
      console.log(`表示圆柱体组件的点云: ${cloudCylinder.size} 个数据点。`);

      return {
        planeCoefficients: coefficientsPlane.values,
        cylinderCoefficients: coefficientsCylinder.values,
        originalSize: cloud.size,
        filteredSize: cloudFiltered.size,
        planeSize: cloudPlane.size,
        cylinderSize: cloudCylinder.size,
      };
    }
  } finally {
    // 清理资源
    pass.manager?.delete();
    ne.manager?.delete();
    seg.manager?.delete();
    extract.manager?.delete();
    extractNormals.manager?.delete();
    tree.manager?.delete();

    cloud.manager?.delete();
    cloudFiltered.manager?.delete();
    cloudNormals.manager?.delete();
    cloudFiltered2.manager?.delete();
    cloudNormals2.manager?.delete();

    inliersPlane.manager?.delete();
    inliersCylinder.manager?.delete();
  }
}
