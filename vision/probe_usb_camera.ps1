param([string]$CameraName = 'USB Camera', [switch]$TestBrightness)
$ErrorActionPreference = 'Stop'
Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.ComTypes;

[ComImport, Guid("29840822-5B84-11D0-BD3B-00A0C911CE86"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface ICreateDevEnum {
    [PreserveSig] int CreateClassEnumerator(ref Guid category, out IEnumMoniker enumerator, int flags);
}
[ComImport, Guid("55272A00-42CB-11CE-8135-00AA004BB851"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IPropertyBag {
    [PreserveSig] int Read([MarshalAs(UnmanagedType.LPWStr)] string name, [MarshalAs(UnmanagedType.Struct)] out object value, IntPtr errorLog);
    [PreserveSig] int Write([MarshalAs(UnmanagedType.LPWStr)] string name, ref object value);
}
[ComImport, Guid("C6E13360-30AC-11D0-A18C-00A0C9118956"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IAMVideoProcAmp {
    [PreserveSig] int GetRange(int property, out int min, out int max, out int step, out int defaultValue, out int caps);
    [PreserveSig] int Set(int property, int value, int flags);
    [PreserveSig] int Get(int property, out int value, out int flags);
}
[ComImport, Guid("C6E13370-30AC-11D0-A18C-00A0C9118956"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IAMCameraControl {
    [PreserveSig] int GetRange(int property, out int min, out int max, out int step, out int defaultValue, out int caps);
    [PreserveSig] int Set(int property, int value, int flags);
    [PreserveSig] int Get(int property, out int value, out int flags);
}
public class CameraProperty {
    public string Group, Name;
    public int Min, Max, Step, Default, Capabilities, Value, Flags, ReadStatus;
}
public class CameraProbeResult {
    public string Name;
    public List<CameraProperty> Controls = new List<CameraProperty>();
    public string BrightnessTest = "Not requested";
}
public static class CameraProbe {
    public static CameraProbeResult Run(string cameraName, bool testBrightness) {
        object enumeratorObject = null, filter = null;
        IEnumMoniker enumerator = null;
        try {
            enumeratorObject = Activator.CreateInstance(Type.GetTypeFromCLSID(new Guid("62BE5D10-60EB-11D0-BD3B-00A0C911CE86")));
            Guid category = new Guid("860BB310-5D01-11D0-BD3B-00A0C911CE86");
            if (((ICreateDevEnum)enumeratorObject).CreateClassEnumerator(ref category, out enumerator, 0) != 0)
                throw new Exception("No video capture devices found.");
            IMoniker[] monikers = new IMoniker[1];
            while (enumerator.Next(1, monikers, IntPtr.Zero) == 0) {
                object bagObject = null;
                try {
                    Guid bagId = typeof(IPropertyBag).GUID;
                    monikers[0].BindToStorage(null, null, ref bagId, out bagObject);
                    object name;
                    ((IPropertyBag)bagObject).Read("FriendlyName", out name, IntPtr.Zero);
                    if (String.Equals((string)name, cameraName, StringComparison.OrdinalIgnoreCase)) {
                        Guid filterId = new Guid("56A86895-0AD4-11CE-B03A-0020AF0BA770");
                        monikers[0].BindToObject(null, null, ref filterId, out filter);
                        break;
                    }
                } finally {
                    if (bagObject != null) Marshal.ReleaseComObject(bagObject);
                    Marshal.ReleaseComObject(monikers[0]);
                }
            }
            if (filter == null) throw new Exception("Requested camera not found: " + cameraName);
            CameraProbeResult result = new CameraProbeResult { Name = cameraName };
            IAMVideoProcAmp video = filter as IAMVideoProcAmp;
            IAMCameraControl camera = filter as IAMCameraControl;
            string[] videoNames = { "Brightness", "Contrast", "Hue", "Saturation", "Sharpness", "Gamma", "ColorEnable", "WhiteBalance", "BacklightCompensation", "Gain" };
            string[] cameraNames = { "Pan", "Tilt", "Roll", "Zoom", "Exposure", "Iris", "Focus" };
            for (int group = 0; group < 2; group++) {
                string[] names = group == 0 ? videoNames : cameraNames;
                for (int i = 0; i < names.Length; i++) {
                    int min, max, step, def, caps, value, flags;
                    int hr;
                    if (group == 0) {
                        if (video == null) continue;
                        hr = video.GetRange(i, out min, out max, out step, out def, out caps);
                    } else {
                        if (camera == null) continue;
                        hr = camera.GetRange(i, out min, out max, out step, out def, out caps);
                    }
                    if (hr < 0) continue;
                    hr = group == 0 ? video.Get(i, out value, out flags) : camera.Get(i, out value, out flags);
                    result.Controls.Add(new CameraProperty { Group = group == 0 ? "VideoProcAmp" : "CameraControl", Name = names[i], Min = min, Max = max, Step = step, Default = def, Capabilities = caps, Value = value, Flags = flags, ReadStatus = hr });
                }
            }
            if (testBrightness) {
                CameraProperty p = result.Controls.Find(x => x.Name == "Brightness");
                if (p == null || p.ReadStatus < 0 || (p.Capabilities & 2) == 0 || p.Step <= 0 || p.Min == p.Max) {
                    result.BrightnessTest = "Manual brightness test unavailable";
                } else {
                    int target = p.Value + p.Step <= p.Max ? p.Value + p.Step : p.Value - p.Step;
                    int setStatus = 0, restoreStatus = 0, readStatus = 0, actual = 0, flags = 0;
                    try {
                        setStatus = video.Set(0, target, 2);
                        readStatus = video.Get(0, out actual, out flags);
                    } finally {
                        restoreStatus = video.Set(0, p.Value, p.Flags);
                    }
                    int restored, restoredFlags;
                    int verifyRestore = video.Get(0, out restored, out restoredFlags);
                    result.BrightnessTest = String.Format("Original={0}; Requested={1}; Readback={2}; SetStatus={3}; ReadStatus={4}; RestoreStatus={5}; Restored={6}; RestoredFlags={7}; RestoreReadStatus={8}", p.Value, target, actual, setStatus, readStatus, restoreStatus, restored, restoredFlags, verifyRestore);
                }
            }
            return result;
        } finally {
            if (filter != null) Marshal.ReleaseComObject(filter);
            if (enumerator != null) Marshal.ReleaseComObject(enumerator);
            if (enumeratorObject != null) Marshal.ReleaseComObject(enumeratorObject);
        }
    }
}
'@
$result = [CameraProbe]::Run($CameraName, $TestBrightness.IsPresent)
$json = $result | ConvertTo-Json -Depth 5
$output = Join-Path (Split-Path $PSScriptRoot -Parent) 'runs\camera_usb_controls.json'
$json | Set-Content -LiteralPath $output -Encoding UTF8
$json
