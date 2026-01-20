#!/usr/bin/env python3
"""
音频格式转换工具
将各种音频格式批量转换为 FLAC 格式
"""

import os
import subprocess
from pathlib import Path
from typing import List, Optional, Union
import argparse


class AudioConverter:
    """音频格式转换器"""
    
    # 支持的音频格式
    SUPPORTED_FORMATS = {'.mp3', '.wav', '.m4a', '.aac', '.ogg', '.opus', 
                        '.wma', '.flac', '.ape', '.ac3', '.aiff'}
    
    def __init__(self, use_ffmpeg: bool = True):
        """
        Args:
            use_ffmpeg: 是否使用 ffmpeg（推荐）
        """
        self.use_ffmpeg = use_ffmpeg
        if use_ffmpeg:
            self._check_ffmpeg()
    
    def _check_ffmpeg(self):
        """检查 ffmpeg 是否安装"""
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         stdout=subprocess.PIPE, 
                         stderr=subprocess.PIPE, 
                         check=True)
            print("✓ ffmpeg 已安装")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "ffmpeg 未安装。请运行: apt-get install -y ffmpeg"
            )
    
    def convert_file(self, 
                    input_path: str, 
                    output_path: str,
                    sample_rate: Optional[int] = None,
                    channels: Optional[int] = None,
                    compression_level: int = 5) -> bool:
        """
        转换单个音频文件到 FLAC
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            sample_rate: 采样率（None=保持原样）
            channels: 声道数（None=保持原样，1=单声道，2=立体声）
            compression_level: FLAC 压缩级别 (0-8, 默认5)
        
        Returns:
            是否转换成功
        """
        if not os.path.exists(input_path):
            print(f"❌ 文件不存在: {input_path}")
            return False
        
        # 如果已经是 FLAC 且不需要重采样/转声道，直接复制
        if (input_path.lower().endswith('.flac') and 
            sample_rate is None and channels is None):
            import shutil
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copy2(input_path, output_path)
            return True
        
        # 构建 ffmpeg 命令
        cmd = ['ffmpeg', '-i', input_path]
        
        # 采样率
        if sample_rate:
            cmd.extend(['-ar', str(sample_rate)])
        
        # 声道数
        if channels:
            cmd.extend(['-ac', str(channels)])
        
        # FLAC 编码器和压缩级别
        cmd.extend([
            '-acodec', 'flac',
            '-compression_level', str(compression_level),
            '-y',  # 覆盖输出文件
            output_path
        ])
        
        try:
            # 创建输出目录
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 执行转换
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ 转换失败: {input_path}")
            print(f"   错误: {e.stderr.decode()}")
            return False
        except Exception as e:
            print(f"❌ 转换失败: {input_path}")
            print(f"   错误: {e}")
            return False
    
    def find_audio_files(self, 
                        root_dir: str, 
                        recursive: bool = True) -> List[str]:
        """
        查找目录中的所有音频文件
        
        Args:
            root_dir: 根目录
            recursive: 是否递归搜索子目录
        
        Returns:
            音频文件路径列表
        """
        audio_files = []
        
        if recursive:
            for root, dirs, files in os.walk(root_dir):
                for file in files:
                    if Path(file).suffix.lower() in self.SUPPORTED_FORMATS:
                        audio_files.append(os.path.join(root, file))
        else:
            for file in os.listdir(root_dir):
                file_path = os.path.join(root_dir, file)
                if os.path.isfile(file_path):
                    if Path(file).suffix.lower() in self.SUPPORTED_FORMATS:
                        audio_files.append(file_path)
        
        return sorted(audio_files)
    
    def convert_all_to_flac(self,
                           input_dir: str,
                           output_dir: str,
                           recursive: bool = True,
                           preserve_structure: bool = True,
                           sample_rate: Optional[int] = None,
                           channels: Optional[int] = None,
                           compression_level: int = 5,
                           skip_existing: bool = True) -> dict:
        """
        批量转换目录中的所有音频文件到 FLAC
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            recursive: 是否递归搜索子目录
            preserve_structure: 是否保留目录结构
            sample_rate: 目标采样率（None=保持原样）
            channels: 目标声道数（None=保持原样）
            compression_level: FLAC 压缩级别 (0-8)
            skip_existing: 是否跳过已存在的文件
        
        Returns:
            统计信息字典
        """
        print(f"\n{'='*60}")
        print(f"批量转换音频到 FLAC")
        print(f"{'='*60}\n")
        
        print(f"输入目录: {input_dir}")
        print(f"输出目录: {output_dir}")
        print(f"递归搜索: {recursive}")
        print(f"保留结构: {preserve_structure}")
        if sample_rate:
            print(f"采样率: {sample_rate} Hz")
        if channels:
            print(f"声道数: {channels}")
        print(f"压缩级别: {compression_level}")
        print()
        
        # 查找所有音频文件
        print("搜索音频文件...")
        audio_files = self.find_audio_files(input_dir, recursive)
        
        if not audio_files:
            print("❌ 未找到音频文件")
            return {'total': 0, 'success': 0, 'failed': 0, 'skipped': 0}
        
        print(f"找到 {len(audio_files)} 个音频文件\n")
        
        # 统计信息
        stats = {
            'total': len(audio_files),
            'success': 0,
            'failed': 0,
            'skipped': 0
        }
        
        # 转换每个文件
        for idx, input_path in enumerate(audio_files, 1):
            # 构建输出路径
            if preserve_structure:
                # 保留目录结构
                rel_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, rel_path)
            else:
                # 扁平化输出
                output_path = os.path.join(output_dir, os.path.basename(input_path))
            
            # 修改扩展名为 .flac
            output_path = os.path.splitext(output_path)[0] + '.flac'
            
            # 检查是否跳过已存在的文件
            if skip_existing and os.path.exists(output_path):
                print(f"[{idx}/{stats['total']}] ⊘ 跳过（已存在）: {os.path.basename(input_path)}")
                stats['skipped'] += 1
                continue
            
            # 转换
            print(f"[{idx}/{stats['total']}] 转换: {os.path.basename(input_path)}")
            
            success = self.convert_file(
                input_path=input_path,
                output_path=output_path,
                sample_rate=sample_rate,
                channels=channels,
                compression_level=compression_level
            )
            
            if success:
                stats['success'] += 1
                print(f"           ✓ 成功 → {os.path.basename(output_path)}")
            else:
                stats['failed'] += 1
        
        # 打印统计信息
        print(f"\n{'='*60}")
        print(f"转换完成")
        print(f"{'='*60}\n")
        print(f"总文件数: {stats['total']}")
        print(f"成功: {stats['success']}")
        print(f"失败: {stats['failed']}")
        print(f"跳过: {stats['skipped']}")
        print()
        
        return stats


def convert_all2flac(input_dir: str,
                    output_dir: str,
                    recursive: bool = True,
                    preserve_structure: bool = True,
                    sample_rate: Optional[int] = None,
                    channels: Optional[int] = None,
                    compression_level: int = 5,
                    skip_existing: bool = True) -> dict:
    """
    批量转换音频文件到 FLAC 格式
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        recursive: 是否递归搜索子目录（默认 True）
        preserve_structure: 是否保留目录结构（默认 True）
        sample_rate: 目标采样率，None=保持原样（例如: 16000, 44100, 48000）
        channels: 目标声道数，None=保持原样（1=单声道, 2=立体声）
        compression_level: FLAC 压缩级别 0-8（默认 5）
        skip_existing: 是否跳过已存在的文件（默认 True）
    
    Returns:
        统计信息字典 {'total': 总数, 'success': 成功, 'failed': 失败, 'skipped': 跳过}
    
    Examples:
        >>> # 基本用法
        >>> convert_all2flac('/input/dir', '/output/dir')
        
        >>> # 转换为 16kHz 单声道
        >>> convert_all2flac('/input', '/output', sample_rate=16000, channels=1)
        
        >>> # 不保留目录结构
        >>> convert_all2flac('/input', '/output', preserve_structure=False)
    """
    converter = AudioConverter(use_ffmpeg=True)
    
    return converter.convert_all_to_flac(
        input_dir=input_dir,
        output_dir=output_dir,
        recursive=recursive,
        preserve_structure=preserve_structure,
        sample_rate=sample_rate,
        channels=channels,
        compression_level=compression_level,
        skip_existing=skip_existing
    )


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='批量转换音频文件到 FLAC 格式'
    )
    
    parser.add_argument(
        'input_dir',
        help='输入目录'
    )
    parser.add_argument(
        'output_dir',
        help='输出目录'
    )
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='不递归搜索子目录'
    )
    parser.add_argument(
        '--no-preserve',
        action='store_true',
        help='不保留目录结构（扁平化输出）'
    )
    parser.add_argument(
        '--sample-rate',
        type=int,
        help='目标采样率（例如: 16000, 44100, 48000）'
    )
    parser.add_argument(
        '--channels',
        type=int,
        choices=[1, 2],
        help='目标声道数（1=单声道, 2=立体声）'
    )
    parser.add_argument(
        '--compression',
        type=int,
        default=5,
        choices=range(9),
        help='FLAC 压缩级别 (0-8, 默认 5)'
    )
    parser.add_argument(
        '--no-skip',
        action='store_true',
        help='不跳过已存在的文件（覆盖）'
    )
    
    args = parser.parse_args()
    
    # 执行转换
    stats = convert_all2flac(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        recursive=not args.no_recursive,
        preserve_structure=not args.no_preserve,
        sample_rate=args.sample_rate,
        channels=args.channels,
        compression_level=args.compression,
        skip_existing=not args.no_skip
    )
    
    # 返回状态码
    return 0 if stats['failed'] == 0 else 1


if __name__ == "__main__":
    import sys
    
    # 示例用法
    if len(sys.argv) == 1:
        print("音频格式转换工具\n")
        print("基本用法:")
        print("  python convert_to_flac.py /input/dir /output/dir\n")
        print("高级用法:")
        print("  # 转换为 16kHz 单声道")
        print("  python convert_to_flac.py /input /output --sample-rate 16000 --channels 1\n")
        print("  # 不保留目录结构")
        print("  python convert_to_flac.py /input /output --no-preserve\n")
        print("  # 不递归搜索")
        print("  python convert_to_flac.py /input /output --no-recursive\n")
        print("Python 函数用法:")
        print("  from convert_to_flac import convert_all2flac")
        print("  convert_all2flac('/input/dir', '/output/dir')")
        sys.exit(0)
    
    sys.exit(main())