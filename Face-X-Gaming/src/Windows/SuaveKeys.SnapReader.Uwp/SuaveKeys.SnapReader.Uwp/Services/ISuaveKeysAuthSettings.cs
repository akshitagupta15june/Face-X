using System;
using System.Collections.Generic;
using System.Text;

namespace SuaveKeys.SnapReader.Uwp.Services
{
    public interface ISuaveKeysAuthSettings
    {
        string ClientId { get; }
        string ClientSecret { get;}
    }
}
